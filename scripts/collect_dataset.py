"""
Скрипт для сбора датасета эмоций через веб-камеру.

Использование:
    python scripts/collect_dataset.py --emotion Happy --output dataset/train/Happy --count 100

Этот скрипт помогает собрать собственный датасет для обучения модели.
Поддерживает сбор для всех 3 эмоций: Happy, Sad, Neutral.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.config import EMOTIONS, TARGET_SIZE


def collect_emotion_data(emotion_name: str, output_dir: str, count: int = 100):
    """
    Собирает изображения эмоции через веб-камеру.
    
    Args:
        emotion_name: Название эмоции (Happy, Sad, Neutral)
        output_dir: Директория для сохранения изображений
        count: Количество изображений для сбора
    """
    # Создаём директорию
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"СБОР ДАТАСЕТА: {emotion_name}")
    print("=" * 60)
    print(f"Выходная директория: {output_dir}")
    print(f"Количество изображений: {count}")
    print()
    print("Инструкции:")
    print(f"  1. Примите выражение лица: {emotion_name.upper()}")
    print("  2. Нажмите 's' для снимка")
    print("  3. Нажмите 'q' для выхода")
    print(f"  4. Собрано: 0/{count}")
    print("-" * 60)
    
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ОШИБКА: Не удалось открыть веб-камеру")
        sys.exit(1)
    
    # Установка разрешения
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    collected = 0
    last_save_time = 0
    save_interval = 0.5  # Минимальный интервал между снимками (сек)
    
    try:
        while collected < count:
            ret, frame = cap.read()
            
            if not ret:
                print("ОШИБКА: Не удалось захватить кадр")
                break
            
            # Отображение информации
            current_time = time.time()
            can_save = (current_time - last_save_time) >= save_interval
            
            # Текст статуса
            status_text = f"Collected: {collected}/{count}"
            if can_save:
                status_color = (0, 255, 0)  # Зелёный — можно снимать
            else:
                status_color = (0, 255, 255)  # Жёлтый — ждём
            
            # Рисуем рамку области захвата
            h, w = frame.shape[:2]
            margin_x, margin_y = int(w * 0.25), int(h * 0.25)
            cv2.rectangle(frame, (margin_x, margin_y), 
                         (w - margin_x, h - margin_y), 
                         (0, 255, 0), 2)
            
            # Отображение текста
            cv2.putText(frame, f"Emotion: {emotion_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, status_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, "'s' - snap, 'q' - quit", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Показ кадра
            cv2.imshow(f"Collecting: {emotion_name}", frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print(f"\nПрервано пользователем. Собрано: {collected}/{count}")
                break
            
            elif key == ord('s') and can_save:
                # Сохраняем центральную область (предполагаем лицо в центре)
                face_region = frame[margin_y:h-margin_y, margin_x:w-margin_x]
                
                # Конвертируем в grayscale и ресайзим до 48x48
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                
                # Генерируем имя файла
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{emotion_name.lower()}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Сохраняем
                cv2.imwrite(filepath, resized)
                collected += 1
                last_save_time = current_time
                
                print(f"✓ Сохранено: {filename} ({collected}/{count})")
                
                # Обновляем статус на экране
                cv2.putText(frame, f"SAVED! {collected}/{count}", 
                           (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(f"Collecting: {emotion_name}", frame)
                cv2.waitKey(200)  # Короткая задержка для визуального подтверждения
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print(f"СБОР ЗАВЕРШЁН")
    print("=" * 60)
    print(f"Сохранено изображений: {collected}")
    print(f"Директория: {output_dir}")
    print()
    
    return collected


def main():
    parser = argparse.ArgumentParser(
        description="Сбор датасета эмоций через веб-камеру",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/collect_dataset.py --emotion Happy --count 100
  python scripts/collect_dataset.py --emotion Sad --output dataset/train/Sad
  python scripts/collect_dataset.py --emotion Neutral --count 50

Советы по сбору:
  - Убедитесь в хорошем освещении
  - Лицо должно быть хорошо видно
  - Избегайте резких движений головы
  - Делайте разные выражения одной эмоции
        """
    )
    
    parser.add_argument(
        "--emotion", "-e",
        type=str,
        required=True,
        choices=EMOTIONS,
        help=f"Название эмоции: {', '.join(EMOTIONS)}"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Директория для сохранения (по умолчанию: dataset/train/{emotion})"
    )
    
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=100,
        help="Количество изображений для сбора (по умолчанию: 100)"
    )
    
    args = parser.parse_args()
    
    # Определяем выходную директорию
    if args.output is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "train")
        output_dir = os.path.join(base_dir, args.emotion)
    else:
        output_dir = args.output
    
    # Запуск сбора
    collected = collect_emotion_data(args.emotion, output_dir, args.count)
    
    if collected == 0:
        print("ПРЕДУПРЕЖДЕНИЕ: Не было собрано ни одного изображения!")
        sys.exit(1)
    
    print("\nРекомендации:")
    print("  1. Проверьте сохранённые изображения")
    print("  2. Удалите некачественные снимки")
    print("  3. Повторите сбор для других эмоций")
    print()
    print(f"Для сбора следующей эмоции выполните:")
    other_emotions = [e for e in EMOTIONS if e != args.emotion]
    for emo in other_emotions:
        print(f"  python scripts/collect_dataset.py --emotion {emo} --count {args.count}")


if __name__ == "__main__":
    main()
