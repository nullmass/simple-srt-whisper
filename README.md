# Listen

Listen is a local Windows application that turns audio and video speech into `.srt` subtitles. It uses `faster-whisper` for speech recognition and has separate CPU and NVIDIA CUDA launch modes.

## How it works

1. Both launchers use one shared `.venv`, created with Python 3.10–3.14 if needed.
2. It downloads FFmpeg Essentials into `tools/ffmpeg/bin/` if FFmpeg is not already present.
3. It installs the shared Python packages from `req.txt`. CUDA mode additionally installs the NVIDIA runtimes from `req-cuda.txt`; PyTorch is not required.
4. `main.py` opens the selected audio or video file, loads the requested Whisper model, transcribes speech, and writes an `.srt` file next to the input media.

The default model is `small`. Models are downloaded automatically by `faster-whisper` into `models/` the first time they are used. The first run needs an internet connection and several gigabytes of free disk space for packages and the model. Later starts reuse the installed packages, FFmpeg, and cached model.

## Hardware

An NVIDIA GPU is **not required**. `start_cpu.bat` explicitly runs on the CPU, including when CUDA packages are installed. `start_cuda.bat` adds CUDA 12.8 and cuDNN 9 runtime libraries to the same `.venv` and explicitly selects CUDA. CUDA mode needs a supported NVIDIA GPU and compatible display driver. A system-wide CUDA Toolkit installation is not required.

## Quick start

1. Install Python 3.10–3.14 if it is not already installed.
2. Double-click `start_cpu.bat` for CPU mode or `start_cuda.bat` for an NVIDIA GPU.
3. On the first run, wait for setup and model downloads to finish.
4. Choose a media file in the file picker. The generated subtitle file is saved beside it.

You can also pass a file path and options:

```bat
start_cpu.bat "C:\media\lecture.webm" --language ru
start_cuda.bat "C:\media\lecture.webm" --language ru
```

If no path is provided, the application opens a file picker. Switching CPU/CUDA reuses the same environment and models. CPU mode leaves optional GPU packages installed. Run `remove_cuda.bat` to uninstall the optional NVIDIA runtimes and legacy PyTorch packages from `.venv` while keeping CPU mode available. CUDA mode reinstalls its runtimes when next selected.

When upgrading an older release, the new launchers use `.venv` and leave old `.venv-cpu` / `.venv-cuda` folders unused. Those old folders can be deleted after verifying the upgrade; keep `models/` to retain downloaded models.

## Features

- Audio and video input, including `.webm`, `.mp4`, `.mkv`, `.mp3`, and `.wav`.
- CPU transcription with `start_cpu.bat`, or CUDA acceleration with `start_cuda.bat` on a supported NVIDIA system.
- Voice activity detection and batched inference.
- Configurable language, model, compute type, and subtitle layout.
- Output subtitles saved next to the input file.

## Common options

```text
--model MODEL              faster-whisper model name or local CTranslate2 model path
--device {auto,cpu,cuda}   explicit device for manual runs; launchers select their mode
--language LANGUAGE        language code, such as ru, en, or ko
--batch-size N             inference batch size (default: 16)
--compute-type TYPE        for example float16, int8, or int8_float16
--no-vad                   disable voice activity detection
--min-silence-ms N         minimum VAD silence duration (default: 500)
--max-chars-per-line N     maximum subtitle line length
--max-lines N              maximum lines per subtitle
--min-sub-duration N       minimum subtitle duration
--max-sub-duration N       maximum subtitle duration
--max-cps N                maximum reading speed in characters per second
```

Example using a larger model:

```bat
start_cpu.bat "C:\media\lecture.mp3" --model medium --language ru
```

## Manual setup

The launchers manage their environment automatically. For manual CPU installation, install `req.txt` in `.venv` and run `main.py --device cpu`. For CUDA, also install `req-cuda.txt`, then run `main.py --device cuda`. The application registers the NVIDIA DLL directories on Windows. No PyTorch installation is needed.

## Project files

- `main.py` — application and subtitle generation pipeline.
- `start_cpu.bat` — one-click CPU environment setup and application launcher.
- `start_cuda.bat` — one-click CUDA environment setup and application launcher.
- `remove_cuda.bat` — optional GPU package removal from the shared environment.
- `setup_and_run.ps1` — shared setup, dependency installation, and launch logic.
- `install_ffmpeg.ps1` — downloads and installs the FFmpeg executables on first run.
- `req.txt` — shared Python application dependencies, including compatibility constraints.
- `req-cuda.txt` — optional NVIDIA runtime dependencies.
- `convert/` — optional scripts for converting legacy Whisper model files.

Downloaded environments, models, FFmpeg binaries, local settings, media, and generated `.srt` files are excluded from Git by `.gitignore`.

## Third-party software and models

FFmpeg is downloaded from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). Python packages and Whisper models are obtained from their configured package and model hosts. Each third-party component and model is governed by its own license; review those terms before redistributing them.

## License

The original code in this project is provided under the MIT License. See [LICENSE](LICENSE). The license disclaimer applies to this project's code; it does not replace the terms or obligations attached to third-party software and models.
