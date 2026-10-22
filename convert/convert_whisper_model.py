"""
Скрипт для конвертации модели Whisper из формата Hugging Face в .pt
"""

import torch
import os
from pathlib import Path

def check_model_files(model_path):
    """Проверяет наличие необходимых файлов модели"""
    model_path = Path(model_path)
    
    print("=" * 60)
    print("Проверка файлов модели")
    print("=" * 60)
    
    required_files = {
        'config.json': model_path / 'config.json',
        'pytorch_model.bin': model_path / 'pytorch_model.bin',
        'model.safetensors': model_path / 'model.safetensors',
    }
    
    found_files = {}
    for name, path in required_files.items():
        exists = path.exists()
        found_files[name] = exists
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {'Найден' if exists else 'Не найден'}")
    
    return found_files

def convert_model(model_path, output_path):
    """Конвертирует модель в формат .pt"""
    
    print("\n" + "=" * 60)
    print("Конвертация модели")
    print("=" * 60)
    
    try:
        from transformers import WhisperForConditionalGeneration
        
        print(f"\nЗагрузка модели из: {model_path}")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
        print(f"Сохранение модели в: {output_path}")
        torch.save(model, output_path)
        
        # Проверяем размер файла
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print("\n" + "=" * 60)
        print("✓ Конвертация завершена успешно!")
        print("=" * 60)
        print(f"Файл: {output_path}")
        print(f"Размер: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Ошибка при конвертации")
        print("=" * 60)
        print(f"Ошибка: {e}")
        return False

def download_model_from_hub(repo_id, local_dir):
    """Скачивает модель с Hugging Face Hub"""
    
    print("\n" + "=" * 60)
    print("Скачивание модели с Hugging Face Hub")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        print(f"\nРепозиторий: {repo_id}")
        print(f"Сохранение в: {local_dir}")
        print("\nЭто может занять несколько минут...\n")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print("\n✓ Модель успешно скачана!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при скачивании: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("КОНВЕРТЕР МОДЕЛЕЙ WHISPER")
    print("=" * 60)
    
    # Путь к вашей модели
    model_path = "./whisper-small-ru"
    
    # Проверяем файлы
    files = check_model_files(model_path)
    
    # Если нет файла с весами
    if not files['pytorch_model.bin'] and not files['model.safetensors']:
        print("\n" + "=" * 60)
        print("⚠️  ВНИМАНИЕ: Файлы с весами модели не найдены!")
        print("=" * 60)
        print("\nВозможные решения:")
        print("\n1. Скачать оригинальную модель Whisper Small:")
        print("   python -c \"from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-small', local_dir='./whisper-small-full')\"")
        print("\n2. Использовать библиотеку openai-whisper:")
        print("   pip install openai-whisper")
        print("   python main.py \"audio.mp3\"")
        print("\n3. Скачать модель вручную с https://huggingface.co/openai/whisper-small")
        
        # Предлагаем скачать
        print("\n" + "=" * 60)
        choice = input("\nСкачать модель openai/whisper-small сейчас? (y/n): ")
        
        if choice.lower() == 'y':
            if download_model_from_hub("openai/whisper-small", "./whisper-small-full"):
                model_path = "./whisper-small-full"
                output_path = "./whisper-small.pt"
                convert_model(model_path, output_path)
        
    else:
        # Конвертируем модель
        output_path = "./whisper-small-ru.pt"
        convert_model(model_path, output_path)

if __name__ == "__main__":
    main()
