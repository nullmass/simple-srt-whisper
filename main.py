import argparse
import json
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

# Keep DLL directory handles alive for delayed CUDA/cuDNN loads on Windows.
_nvidia_dll_directories = []
if os.name == "nt":
    for _package in ("cublas", "cudnn", "cuda_runtime"):
        _bin_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / _package / "bin"
        if _bin_dir.is_dir():
            os.environ["PATH"] = str(_bin_dir) + os.pathsep + os.environ.get("PATH", "")
            _nvidia_dll_directories.append(os.add_dll_directory(str(_bin_dir)))

import ctranslate2
from faster_whisper import BatchedInferencePipeline, WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


AUDIO_AND_VIDEO_FILETYPES = [
    (
        "Медиа файлы",
        "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v *.mpg *.mpeg *.3gp *.ogv "
        "*.mp3 *.wav *.m4a *.flac *.aac *.ogg *.wma *.opus",
    ),
    ("Все файлы", "*.*"),
]


model = None
CONFIG_FILE_NAME = "listen_config.json"
DEFAULT_MAX_CHARS_PER_LINE = 42
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_SUB_DURATION = 1.3
DEFAULT_TARGET_SUB_DURATION = 2.2
DEFAULT_MAX_SUB_DURATION = 4.0
DEFAULT_TARGET_CHARS = 36
DEFAULT_MAX_CPS = 18.0
DEFAULT_TARGET_CPS = 17.0
DEFAULT_MIN_WORDS = 3
DEFAULT_SOFT_PAUSE_MS = 280
DEFAULT_STRONG_PAUSE_MS = 450
LANGUAGE_OPTIONS = {
    "1": ("ru", "Русский"),
    "2": ("en", "Английский"),
    "3": ("ko", "Корейский"),
    "4": ("ja", "Японский"),
    "5": ("zh", "Китайский"),
    "0": (None, "Автоопределение"),
}
FALLBACK_BREAK_WORDS = {
    "и",
    "а",
    "но",
    "или",
    "что",
    "потому",
    "потому что",
    "если",
    "когда",
    "чтобы",
}
PREPOSITIONS = {
    "в",
    "во",
    "на",
    "с",
    "со",
    "к",
    "ко",
    "по",
    "о",
    "об",
    "обо",
    "от",
    "до",
    "из",
    "у",
    "для",
    "про",
    "под",
    "над",
    "при",
    "без",
    "через",
    "между",
}
STRONG_BREAK_PUNCTUATION = (".", "!", "?", "…")
SOFT_BREAK_PUNCTUATION = (",", ";", ":", "—", "-")


def load_config(config_path):
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(config_path, config):
    try:
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
    except OSError:
        pass


def get_initial_dir(script_dir):
    config_path = os.path.join(script_dir, CONFIG_FILE_NAME)
    config = load_config(config_path)
    last_dir = config.get("last_dir")

    if last_dir and os.path.isdir(last_dir):
        return last_dir

    return script_dir


def update_last_dir(script_dir, file_path):
    selected_dir = os.path.dirname(file_path)
    if not selected_dir:
        return

    config_path = os.path.join(script_dir, CONFIG_FILE_NAME)
    config = load_config(config_path)
    config["last_dir"] = selected_dir
    save_config(config_path, config)


def select_file_via_explorer(initial_dir=None):
    """Открывает проводник и возвращает путь к выбранному файлу."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Выберите аудио/видео файл",
        initialdir=initial_dir or os.getcwd(),
        filetypes=AUDIO_AND_VIDEO_FILETYPES,
    )
    root.destroy()
    return file_path


def select_language_interactive():
    print("Выберите язык распознавания:")
    for key, (code, label) in LANGUAGE_OPTIONS.items():
        print(f"  {key}. {label}{f' ({code})' if code else ''}")
    print("Можно также ввести код языка вручную, например: ko, ru, en")

    selected = input("Язык [ru]: ").strip().lower()
    if not selected:
        return "ru"

    if selected in LANGUAGE_OPTIONS:
        return LANGUAGE_OPTIONS[selected][0]

    if selected in {"auto", "detect", "none"}:
        return None

    return selected


def detect_device():
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def default_compute_type(device):
    return "float16" if device == "cuda" else "int8"


def resolve_model_source(model_arg, script_dir):
    model_path = Path(model_arg)
    if model_path.is_file():
        if model_path.suffix.lower() == ".pt":
            raise ValueError(
                "faster-whisper не поддерживает .pt-модели из openai-whisper. "
                "Передайте имя модели faster-whisper или путь к папке CTranslate2."
            )
        raise ValueError("Ожидалось имя модели или путь к директории модели, а не файл.")

    if model_path.is_dir():
        return str(model_path)

    local_model_dir = Path(script_dir) / "models" / model_arg
    if local_model_dir.is_dir():
        return str(local_model_dir)

    return model_arg


def load_model_once(model_name, device, compute_type, download_root):
    global model
    if model is None:
        print(f"Загрузка модели faster-whisper: {model_name}")
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
    return model


def transcribe_media(
    media_path,
    model_instance,
    batch_size,
    language,
    vad_filter,
    min_silence_duration_ms,
):
    vad_parameters = {"min_silence_duration_ms": min_silence_duration_ms}

    if batch_size > 1:
        pipeline = BatchedInferencePipeline(model=model_instance)
        return pipeline.transcribe(
            media_path,
            language=language,
            batch_size=batch_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=True,
            beam_size=1,
            condition_on_previous_text=False,
        )

    return model_instance.transcribe(
        media_path,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        word_timestamps=True,
        beam_size=1,
        condition_on_previous_text=False,
    )


def format_time(seconds):
    """Преобразует время в формате секунд в формат SRT."""
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))

    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0

    minutes, seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def normalize_word_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_caption_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+-(?=\w)", "-", text)
    return text


def raw_token_text(word):
    return normalize_word_text(word["text"])


def token_for_rules(word):
    return re.sub(r"^[^\wА-Яа-яЁё]+|[^\wА-Яа-яЁё]+$", "", raw_token_text(word)).strip()


def is_numeric_like(token):
    return bool(re.fullmatch(r"[\d]+", token))


def is_capitalized_word(token):
    if not token:
        return False
    return token[0].isupper() and token[1:].islower()


def forms_protected_pair(left_word, right_word):
    left = token_for_rules(left_word)
    right = token_for_rules(right_word)
    if not left or not right:
        return False

    if left.lower() in PREPOSITIONS:
        return True

    if is_numeric_like(left) or is_numeric_like(right):
        return True

    if left.lower() in {"vue", "react", "python", "java", "c", "csharp", "c++", "it"}:
        return True

    if is_capitalized_word(left) and is_capitalized_word(right):
        return True

    return False


def build_line_text(words):
    return normalize_caption_text(" ".join(word["text"] for word in words))


def split_text_evenly(words, max_chars_per_line, max_lines):
    text = build_line_text(words)
    tokens = text.split()
    if not tokens:
        return None

    if len(text) <= max_chars_per_line:
        return [text]

    if max_lines < 2:
        return None

    best_lines = None
    best_score = None

    for split_index in range(1, len(words)):
        if forms_protected_pair(words[split_index - 1], words[split_index]):
            continue

        first = build_line_text(words[:split_index])
        second = build_line_text(words[split_index:])

        if not first or not second:
            continue

        if len(first) > max_chars_per_line or len(second) > max_chars_per_line:
            continue

        score = abs(len(first) - len(second))
        # Слегка предпочитаем нижнюю строку не длиннее верхней.
        if len(second) > len(first):
            score += 5

        if best_score is None or score < best_score:
            best_score = score
            best_lines = [first, second]

    return best_lines


def build_caption_text(words, max_chars_per_line, max_lines):
    text = build_line_text(words)
    if not text:
        return None

    lines = split_text_evenly(words, max_chars_per_line, max_lines)
    if lines is None:
        if len(text) <= max_chars_per_line:
            lines = [text]
        else:
            return None

    return "\n".join(lines)


def caption_duration(words):
    if not words:
        return 0.0
    return words[-1]["end"] - words[0]["start"]


def caption_char_count(words):
    return len(build_line_text(words))


def estimate_cps(words):
    if not words:
        return 0.0

    duration = max(caption_duration(words), 0.01)
    return caption_char_count(words) / duration


def should_break_after(words, min_duration):
    if not words:
        return False

    text = normalize_caption_text(" ".join(word["text"] for word in words))
    duration = words[-1]["end"] - words[0]["start"]
    last_token = words[-1]["text"]

    if duration < min_duration:
        return False

    if last_token.endswith(STRONG_BREAK_PUNCTUATION):
        return True

    if last_token.endswith(SOFT_BREAK_PUNCTUATION) and duration >= min_duration * 1.2:
        return True

    return False


def word_gap(previous_word, current_word):
    return max(0.0, current_word["start"] - previous_word["end"])


def segment_to_word_items(segment):
    word_items = []
    words = getattr(segment, "words", None) or []

    for word in words:
        text = normalize_word_text(getattr(word, "word", ""))
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)

        if not text or start is None or end is None or end <= start:
            continue

        word_items.append({"text": text, "start": start, "end": end})

    if word_items:
        return word_items

    text = normalize_caption_text(getattr(segment, "text", ""))
    if not text:
        return []

    tokens = text.split()
    if not tokens:
        return []

    segment_duration = max(segment.end - segment.start, 0.01)
    token_duration = segment_duration / len(tokens)
    result = []

    for index, token in enumerate(tokens):
        start = segment.start + token_duration * index
        end = segment.start + token_duration * (index + 1)
        result.append({"text": token, "start": start, "end": end})

    return result


def split_priority(words, split_index, soft_pause_s, strong_pause_s):
    if split_index <= 0 or split_index >= len(words):
        return None

    left_word = words[split_index - 1]
    right_word = words[split_index]

    if forms_protected_pair(left_word, right_word):
        return None

    left_token = raw_token_text(left_word)
    right_token = token_for_rules(right_word).lower()
    gap = word_gap(left_word, right_word)

    if left_token.endswith(STRONG_BREAK_PUNCTUATION):
        return 0
    if left_token.endswith(SOFT_BREAK_PUNCTUATION):
        return 1
    if gap >= strong_pause_s:
        return 2
    if gap >= soft_pause_s:
        return 3
    if right_token in FALLBACK_BREAK_WORDS:
        return 4
    return None


def choose_best_split(
    words,
    candidate_indices,
    max_chars_per_line,
    max_lines,
    min_duration,
    max_duration,
    max_cps,
    target_duration,
    target_chars,
    target_cps,
    soft_pause_s,
    strong_pause_s,
):
    best_choice = None

    for split_index in candidate_indices:
        priority = split_priority(words, split_index, soft_pause_s, strong_pause_s)
        if priority is None:
            continue

        left = words[:split_index]
        right = words[split_index:]
        left_text = build_caption_text(left, max_chars_per_line, max_lines)
        right_text = build_caption_text(right, max_chars_per_line, max_lines) if right else ""
        if left_text is None or (right and right_text is None):
            continue

        left_duration = caption_duration(left)
        left_chars = caption_char_count(left)
        left_cps = estimate_cps(left)

        if left_duration < min_duration:
            continue
        if left_duration > max_duration:
            continue
        if left_cps > max_cps:
            continue

        score = priority * 100
        score += abs(left_duration - target_duration) * 25
        score += abs(left_chars - target_chars) * 2
        score += abs(left_cps - target_cps) * 12

        if best_choice is None or score < best_choice[0]:
            best_choice = (score, split_index)

    return best_choice[1] if best_choice else None


def force_split_index(
    words,
    max_chars_per_line,
    max_lines,
    min_duration,
    max_duration,
    max_cps,
    target_duration,
    target_chars,
    target_cps,
    soft_pause_s,
    strong_pause_s,
):
    candidate_indices = list(range(1, len(words)))
    split_index = choose_best_split(
        words,
        candidate_indices,
        max_chars_per_line,
        max_lines,
        min_duration,
        max_duration,
        max_cps,
        target_duration,
        target_chars,
        target_cps,
        soft_pause_s,
        strong_pause_s,
    )
    if split_index is not None:
        return split_index

    fallback = max(min(len(words) - DEFAULT_MIN_WORDS, len(words) // 2), DEFAULT_MIN_WORDS)
    if fallback >= len(words):
        fallback = len(words) - 1
    if fallback <= 0:
        return None
    return fallback


def score_merge_candidate(words, target_duration, target_chars, target_cps, max_cps):
    duration = caption_duration(words)
    chars = caption_char_count(words)
    cps = estimate_cps(words)
    return (
        abs(duration - target_duration) * 25
        + abs(chars - target_chars) * 2
        + abs(cps - target_cps) * 12
        + max(0.0, cps - max_cps) * 100
    )


def split_into_caption_chunks(
    segments,
    max_chars_per_line,
    max_lines,
    min_duration,
    max_duration,
    max_cps,
):
    words = []
    for segment in segments:
        words.extend(segment_to_word_items(segment))

    target_duration = DEFAULT_TARGET_SUB_DURATION
    target_chars = DEFAULT_TARGET_CHARS
    target_cps = DEFAULT_TARGET_CPS
    min_words = DEFAULT_MIN_WORDS
    soft_pause_s = DEFAULT_SOFT_PAUSE_MS / 1000
    strong_pause_s = DEFAULT_STRONG_PAUSE_MS / 1000
    captions = []

    current = []
    candidate_indices = []

    def flush_current():
        nonlocal current, candidate_indices
        if current:
            captions.append(current)
        current = []
        candidate_indices = []

    for word in words:
        current.append(word)
        current_duration = caption_duration(current)
        current_chars = caption_char_count(current)
        current_cps = estimate_cps(current)
        current_text = build_caption_text(current, max_chars_per_line, max_lines)

        if len(current) >= 2:
            split_index = len(current) - 1
            if split_priority(current, split_index, soft_pause_s, strong_pause_s) is not None:
                candidate_indices.append(split_index)

        if current_text is None:
            split_index = force_split_index(
                current,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
                min_duration=min_duration,
                max_duration=max_duration,
                max_cps=max_cps,
                target_duration=target_duration,
                target_chars=target_chars,
                target_cps=target_cps,
                soft_pause_s=soft_pause_s,
                strong_pause_s=strong_pause_s,
            )
            if split_index is not None:
                captions.append(current[:split_index])
                current = current[split_index:]
                candidate_indices = [
                    idx - split_index for idx in candidate_indices if idx > split_index
                ]
            continue

        if current_duration < min_duration:
            continue

        if (
            current_duration >= target_duration
            or current_chars >= target_chars
            or current_cps > target_cps
        ):
            split_index = choose_best_split(
                current,
                candidate_indices,
                max_chars_per_line,
                max_lines,
                min_duration,
                max_duration,
                max_cps,
                target_duration,
                target_chars,
                target_cps,
                soft_pause_s,
                strong_pause_s,
            )
            if split_index is not None:
                captions.append(current[:split_index])
                current = current[split_index:]
                candidate_indices = [
                    idx - split_index for idx in candidate_indices if idx > split_index
                ]
                continue

        if current_chars > max_chars_per_line * max_lines or current_duration > max_duration:
            split_index = force_split_index(
                current,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
                min_duration=min_duration,
                max_duration=max_duration,
                max_cps=max_cps,
                target_duration=target_duration,
                target_chars=target_chars,
                target_cps=target_cps,
                soft_pause_s=soft_pause_s,
                strong_pause_s=strong_pause_s,
            )
            if split_index is not None:
                captions.append(current[:split_index])
                current = current[split_index:]
                candidate_indices = [
                    idx - split_index for idx in candidate_indices if idx > split_index
                ]

    flush_current()

    return merge_short_caption_chunks(
        [chunk for chunk in captions if chunk],
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
        min_words=min_words,
        target_duration=target_duration,
        target_chars=target_chars,
        target_cps=target_cps,
    )


def merge_short_caption_chunks(
    captions,
    max_chars_per_line,
    max_lines,
    min_duration,
    max_duration,
    max_cps,
    min_words,
    target_duration,
    target_chars,
    target_cps,
):
    if not captions:
        return captions

    merged = [chunk for chunk in captions if chunk]
    changed = True

    while changed:
        changed = False
        index = 0

        while index < len(merged):
            current = merged[index]
            current_duration = caption_duration(current)
            current_chars = caption_char_count(current)

            if (
                current_duration >= min_duration
                and current_chars >= 18
                and len(current) >= min_words
            ):
                index += 1
                continue

            choices = []

            if index > 0:
                prev_candidate = merged[index - 1] + current
                if (
                    build_caption_text(prev_candidate, max_chars_per_line, max_lines) is not None
                    and caption_duration(prev_candidate) <= max_duration
                    and estimate_cps(prev_candidate) <= max_cps
                ):
                    choices.append(
                        (
                            score_merge_candidate(
                                prev_candidate,
                                target_duration,
                                target_chars,
                                target_cps,
                                max_cps,
                            ),
                            "prev",
                        )
                    )

            if index + 1 < len(merged):
                next_candidate = current + merged[index + 1]
                if (
                    build_caption_text(next_candidate, max_chars_per_line, max_lines) is not None
                    and caption_duration(next_candidate) <= max_duration
                    and estimate_cps(next_candidate) <= max_cps
                ):
                    choices.append(
                        (
                            score_merge_candidate(
                                next_candidate,
                                target_duration,
                                target_chars,
                                target_cps,
                                max_cps,
                            ),
                            "next",
                        )
                    )

            if not choices:
                index += 1
                continue

            choices.sort(key=lambda item: item[0])
            direction = choices[0][1]

            if direction == "prev":
                merged[index - 1] = merged[index - 1] + current
                del merged[index]
                index = max(0, index - 1)
            else:
                merged[index] = current + merged[index + 1]
                del merged[index + 1]

            changed = True

    return merged


def create_srt_from_segments(
    segments,
    output_srt_path,
    max_chars_per_line,
    max_lines,
    min_duration,
    max_duration,
    max_cps,
):
    """Создает файл субтитров SRT из сегментов faster-whisper."""
    caption_chunks = split_into_caption_chunks(
        segments,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
    )

    with open(output_srt_path, "w", encoding="utf-8") as file:
        idx = 1
        previous_text = ""

        for chunk in caption_chunks:
            text = build_caption_text(chunk, max_chars_per_line, max_lines)

            if not text or text == previous_text:
                continue

            start_time = chunk[0]["start"]
            end_time = chunk[-1]["end"]

            if end_time - start_time < 0.3:
                continue

            file.write(f"{idx}\n")
            file.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            file.write(f"{text}\n\n")

            previous_text = text
            idx += 1


def save_subtitles_near_media(
    original_path,
    segments,
    max_chars_per_line,
    max_lines,
    min_duration,
    max_duration,
    max_cps,
):
    """Сохраняет субтитры рядом с исходным файлом."""
    media_dir = os.path.dirname(original_path)
    media_name = os.path.splitext(os.path.basename(original_path))[0]
    subtitle_path = os.path.join(media_dir, f"{media_name}.srt")

    create_srt_from_segments(
        segments,
        subtitle_path,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
    )
    print(f"Субтитры сохранены: {subtitle_path}")


def transcribe_with_progress(
    media_path,
    model_instance,
    batch_size,
    language,
    vad_filter,
    min_silence_duration_ms,
):
    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Распознавание речи...", total=None)
        segments_iter, info = transcribe_media(
            media_path,
            model_instance,
            batch_size,
            language,
            vad_filter,
            min_silence_duration_ms,
        )
        segments = list(segments_iter)
        progress.remove_task(task)

    return segments, info


def main():
    parser = argparse.ArgumentParser(
        description="Распознавание речи и создание субтитров (SRT) на faster-whisper"
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Путь к аудио/видео файлу (если не указан — откроется проводник)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        help="Имя модели faster-whisper или путь к папке модели CTranslate2",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Язык распознавания, например ru, en, ko. Если не указан, будет предложен выбор.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Размер батча для BatchedInferencePipeline (по умолчанию: 16)",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Тип вычислений faster-whisper, например float16, int8, int8_float16",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Отключить VAD-фильтрацию перед распознаванием",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=500,
        help="Минимальная длительность тишины для VAD в миллисекундах",
    )
    parser.add_argument(
        "--max-chars-per-line",
        type=int,
        default=DEFAULT_MAX_CHARS_PER_LINE,
        help="Максимум символов в строке субтитра (по умолчанию: 42)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help="Максимум строк в субтитре (по умолчанию: 2)",
    )
    parser.add_argument(
        "--min-sub-duration",
        type=float,
        default=DEFAULT_MIN_SUB_DURATION,
        help="Минимальная целевая длительность субтитра в секундах (по умолчанию: 1.0)",
    )
    parser.add_argument(
        "--max-sub-duration",
        type=float,
        default=DEFAULT_MAX_SUB_DURATION,
        help="Максимальная длительность субтитра в секундах (по умолчанию: 4.0)",
    )
    parser.add_argument(
        "--max-cps",
        type=float,
        default=DEFAULT_MAX_CPS,
        help="Максимум символов в секунду для субтитра (по умолчанию: 18)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device; the launchers explicitly select cpu or cuda",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_chars_per_line < 10 or args.max_lines < 1:
        print("ОШИБКА: проверьте --batch-size, --max-chars-per-line и --max-lines")
        return

    if args.min_sub_duration <= 0 or args.max_sub_duration <= 0 or args.max_cps <= 0:
        print("ОШИБКА: параметры длительности и --max-cps должны быть > 0")
        return

    if args.min_sub_duration > args.max_sub_duration:
        print("ОШИБКА: --min-sub-duration не может быть больше --max-sub-duration")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = args.file_path or select_file_via_explorer(
        initial_dir=get_initial_dir(script_dir)
    )

    if not file_path:
        print("Файл не выбран. Выход.")
        return

    if not os.path.exists(file_path):
        print(f"ОШИБКА: Файл не найден: {file_path}")
        return

    update_last_dir(script_dir, file_path)

    language = args.language.strip().lower() if args.language else select_language_interactive()
    device = detect_device() if args.device == "auto" else args.device
    compute_type = args.compute_type or default_compute_type(device)
    vad_filter = not args.no_vad

    print(f"Используется устройство: {device}")
    print(f"Тип вычислений: {compute_type}")
    print(f"VAD: {'включен' if vad_filter else 'выключен'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Язык распознавания: {language or 'автоопределение'}")

    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    try:
        model_source = resolve_model_source(args.model, script_dir)
        model_instance = load_model_once(model_source, device, compute_type, models_dir)
        print("Модель загружена успешно")
    except Exception as e:
        print(f"ОШИБКА при загрузке модели: {e}")
        return

    print(f"Распознавание речи из: {file_path}")

    try:
        segments, info = transcribe_with_progress(
            file_path,
            model_instance,
            args.batch_size,
            language,
            vad_filter,
            args.min_silence_ms,
        )
    except Exception as e:
        print(f"ОШИБКА при распознавании: {e}")
        return

    if not segments:
        print("Речь не обнаружена или результат пуст.")
        return

    save_subtitles_near_media(
        file_path,
        segments,
        max_chars_per_line=args.max_chars_per_line,
        max_lines=args.max_lines,
        min_duration=args.min_sub_duration,
        max_duration=args.max_sub_duration,
        max_cps=args.max_cps,
    )

    if getattr(info, "language", None):
        print(f"Определенный язык: {info.language}")
    if getattr(info, "duration", None):
        print(f"Длительность аудио: {info.duration:.2f} сек")

    print("Готово!")


if __name__ == "__main__":
    main()
