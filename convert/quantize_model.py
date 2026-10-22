"""
Программа для квантизации моделей PyTorch с различными пресетами
Поддерживает разные уровни сжатия от минимального до ультра
"""

import torch
import argparse
import os
from pathlib import Path

# Определение пресетов квантизации
PRESETS = {
    'minimal': {
        'name': 'Минимальная квантизация',
        'description': 'Небольшое сжатие, максимальное качество',
        'dtype': torch.qint8,
        'layers': ['Linear'],
        'compression': '~25%'
    },
    'light': {
        'name': 'Легкая квантизация',
        'description': 'Умеренное сжатие, хорошее качество',
        'dtype': torch.qint8,
        'layers': ['Linear', 'Conv1d'],
        'compression': '~40%'
    },
    'medium': {
        'name': 'Средняя квантизация',
        'description': 'Баланс между размером и качеством',
        'dtype': torch.qint8,
        'layers': ['Linear', 'Conv1d', 'Conv2d'],
        'compression': '~50%'
    },
    'high': {
        'name': 'Высокая квантизация',
        'description': 'Сильное сжатие, приемлемое качество',
        'dtype': torch.qint8,
        'layers': ['Linear', 'Conv1d', 'Conv2d', 'LSTM'],
        'compression': '~60%'
    },
    'ultra': {
        'name': 'Ультра квантизация',
        'description': 'Максимальное сжатие, минимальный размер',
        'dtype': torch.qint8,
        'layers': ['Linear', 'Conv1d', 'Conv2d', 'LSTM', 'GRU'],
        'compression': '~75%'
    }
}

def print_header():
    """Выводит заголовок программы"""
    print("\n" + "=" * 70)
    print("  КВАНТИЗАЦИЯ МОДЕЛЕЙ PYTORCH")
    print("  Сжатие моделей для уменьшения размера и ускорения работы")
    print("=" * 70)

def print_presets():
    """Выводит информацию о доступных пресетах"""
    print("\n📋 Доступные пресеты квантизации:\n")
    for preset_id, preset in PRESETS.items():
        print(f"  [{preset_id.upper()}]")
        print(f"  Название: {preset['name']}")
        print(f"  Описание: {preset['description']}")
        print(f"  Сжатие: {preset['compression']}")
        print(f"  Слои: {', '.join(preset['layers'])}")
        print()

def get_file_size_mb(file_path):
    """Возвращает размер файла в МБ"""
    return os.path.getsize(file_path) / (1024 * 1024)

def get_layer_types(layer_names):
    """Преобразует названия слоев в типы PyTorch"""
    import torch.nn as nn
    
    layer_map = {
        'Linear': nn.Linear,
        'Conv1d': nn.Conv1d,
        'Conv2d': nn.Conv2d,
        'LSTM': nn.LSTM,
        'GRU': nn.GRU,
    }
    
    return {layer_map[name] for name in layer_names if name in layer_map}

def quantize_model(model_path, output_path, preset_name):
    """
    Квантизирует модель согласно выбранному пресету
    
    Args:
        model_path: Путь к исходной модели
        output_path: Путь для сохранения квантизированной модели
        preset_name: Название пресета
    """
    
    if preset_name not in PRESETS:
        raise ValueError(f"Неизвестный пресет: {preset_name}")
    
    preset = PRESETS[preset_name]
    
    print("\n" + "=" * 70)
    print(f"  КВАНТИЗАЦИЯ: {preset['name']}")
    print("=" * 70)
    
    # Загрузка модели
    print(f"\n📥 Загрузка модели из: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    
    original_size = get_file_size_mb(model_path)
    print(f"   Размер исходной модели: {original_size:.2f} MB")
    
    # Загружаем модель
    print("\n⚙️  Загрузка модели в память...")
    # Используем weights_only=False для загрузки моделей с классами (например, Whisper)
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    model.eval()  # Переводим в режим inference
    
    # Получаем типы слоев для квантизации
    layer_types = get_layer_types(preset['layers'])
    
    print(f"\n🔧 Применение квантизации...")
    print(f"   Пресет: {preset['name']}")
    print(f"   Тип данных: {preset['dtype']}")
    print(f"   Квантизируемые слои: {', '.join(preset['layers'])}")
    
    # Применяем динамическую квантизацию
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        layer_types,
        dtype=preset['dtype']
    )
    
    # Сохраняем квантизированную модель
    print(f"\n💾 Сохранение квантизированной модели...")
    torch.save(quantized_model, output_path)
    
    quantized_size = get_file_size_mb(output_path)
    compression_ratio = ((original_size - quantized_size) / original_size) * 100
    
    # Результаты
    print("\n" + "=" * 70)
    print("  ✅ КВАНТИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 70)
    print(f"\n📊 Результаты:")
    print(f"   Исходный размер:      {original_size:.2f} MB")
    print(f"   Размер после сжатия:  {quantized_size:.2f} MB")
    print(f"   Сжатие:               {compression_ratio:.1f}%")
    print(f"   Экономия:             {original_size - quantized_size:.2f} MB")
    print(f"\n📁 Файл сохранен: {output_path}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Квантизация моделей PyTorch с различными пресетами',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python quantize_model.py whisper-small-ru.pt --preset minimal
  python quantize_model.py whisper-small-ru.pt --preset medium -o model_medium.pt
  python quantize_model.py whisper-small-ru.pt --preset ultra
  python quantize_model.py whisper-small-ru.pt --list-presets
        """
    )
    
    parser.add_argument(
        'model_path',
        nargs='?',
        help='Путь к файлу модели (.pt)'
    )
    
    parser.add_argument(
        '--preset', '-p',
        choices=list(PRESETS.keys()),
        default='medium',
        help='Пресет квантизации (по умолчанию: medium)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Путь для сохранения квантизированной модели (по умолчанию: добавляется суффикс)'
    )
    
    parser.add_argument(
        '--list-presets', '-l',
        action='store_true',
        help='Показать список доступных пресетов'
    )
    
    args = parser.parse_args()
    
    print_header()
    
    # Если запрошен список пресетов
    if args.list_presets:
        print_presets()
        return
    
    # Проверка наличия пути к модели
    if not args.model_path:
        parser.print_help()
        print("\n❌ Ошибка: Укажите путь к модели")
        print("\nПример: python quantize_model.py whisper-small-ru.pt --preset medium")
        return
    
    # Формируем путь для выходного файла
    if args.output:
        output_path = args.output
    else:
        model_path = Path(args.model_path)
        output_path = model_path.parent / f"{model_path.stem}_{args.preset}{model_path.suffix}"
    
    try:
        # Выполняем квантизацию
        quantize_model(args.model_path, str(output_path), args.preset)
        
        print("\n💡 Совет: Протестируйте квантизированную модель перед использованием")
        print("   Проверьте качество распознавания на тестовом аудио")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("  ❌ ОШИБКА ПРИ КВАНТИЗАЦИИ")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {e}")
        print("\n💡 Возможные решения:")
        print("   1. Убедитесь, что файл модели существует")
        print("   2. Проверьте, что модель в формате .pt")
        print("   3. Убедитесь, что достаточно оперативной памяти")
        print()

if __name__ == "__main__":
    main()
