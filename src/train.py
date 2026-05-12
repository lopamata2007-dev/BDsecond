"""
Скрипт обучения модели распознавания эмоций.

Использует ImageDataGenerator для загрузки данных из структуры папок:
dataset/train/
    Happy/
        image1.png
        image2.png
        ...
    Sad/
        image1.png
        ...
    Neutral/
        image1.png
        ...
dataset/test/
    Happy/
        ...
    Sad/
        ...
    Neutral/
        ...

Адаптация под 3 класса:
- class_mode="categorical" для one-hot encoded меток
- target_size=(48, 48), color_mode="grayscale"
- Финальный слой модели: Dense(3, softmax)
"""

import os
import sys

# Добавляем корень проекта в path для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    MODEL_PATH,
    TARGET_SIZE,
    COLOR_MODE,
    BATCH_SIZE,
    EPOCHS,
    NUM_CLASSES,
    EMOTIONS,
)
from src.model import create_model


def create_data_generators():
    """
    Создаёт генераторы данных для обучения и валидации.
    
    Использует ImageDataGenerator с flow_from_directory для:
    - Автоматической загрузки изображений из папок
    - Нормализации пикселей в [0, 1] (rescale=1./255)
    - One-hot encoding меток (class_mode="categorical")
    
    Адаптация под 3 класса:
    - Классы определяются по названиям подпапок (Happy, Sad, Neutral)
    - NUM_CLASSES = 3 соответствует количеству подпапок
    
    Returns:
        train_generator, validation_generator
    """
    # =====================================================================
    # DATA AUGMENTATION ДЛЯ ОБУЧЕНИЯ
    # =====================================================================
    # Аугментация помогает предотвратить переобучение и улучшает обобщающую способность
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Нормализация пикселей в [0, 1]
        rotation_range=10,  # Случайные повороты до 10 градусов
        width_shift_range=0.1,  # Сдвиг по ширине
        height_shift_range=0.1,  # Сдвиг по высоте
        shear_range=0.1,  # Сдвиг (shear transformation)
        zoom_range=0.1,  # Случайное приближение/отдаление
        horizontal_flip=True,  # Горизонтальное отражение (лицо симметрично)
        fill_mode='nearest'  # Заполнение границ после трансформаций
    )
    
    # =====================================================================
    # ТОЛЬКО НОРМАЛИЗАЦИЯ ДЛЯ ВАЛИДАЦИИ/ТЕСТА
    # =====================================================================
    # Для валидации не применяем аугментацию — только нормализация
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # =====================================================================
    # ЗАГРУЗКА ДАННЫХ ИЗ ПАПОК
    # =====================================================================
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    print(f"Директория обучения: {TRAIN_DIR}")
    print(f"Директория валидации: {TEST_DIR}")
    print(f"Ожидаемые классы ({NUM_CLASSES}): {EMOTIONS}")
    print()
    
    # Генератор для обучающих данных
    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=TARGET_SIZE,  # (48, 48)
        color_mode=COLOR_MODE,  # "grayscale" — 1 канал
        batch_size=BATCH_SIZE,
        class_mode="categorical",  # One-hot encoded метки для 3 классов
        shuffle=True,  # Перемешивание данных
        seed=42  # Воспроизводимость
    )
    
    # Генератор для валидационных данных
    validation_generator = val_datagen.flow_from_directory(
        directory=TEST_DIR,
        target_size=TARGET_SIZE,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False  # Не перемешиваем для корректной оценки
    )
    
    # Проверка соответствия количества классов
    actual_classes = train_generator.class_indices
    print(f"Обнаруженные классы: {list(actual_classes.keys())}")
    print(f"Количество классов: {len(actual_classes)}")
    
    if len(actual_classes) != NUM_CLASSES:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Ожидалось {NUM_CLASSES} классов, найдено {len(actual_classes)}")
        print("Проверьте структуру папок dataset/train и dataset/test")
    
    # Вывод информации о данных
    print()
    print(f"Обучающие примеры: {train_generator.samples}")
    print(f"Валидационные примеры: {validation_generator.samples}")
    print(f"Размер батча: {BATCH_SIZE}")
    print(f"Шагов за эпоху: {train_generator.samples // BATCH_SIZE + 1}")
    print()
    
    return train_generator, validation_generator


def create_callbacks():
    """
    Создаёт callback-функции для контроля процесса обучения.
    
    Callbacks:
    1. ModelCheckpoint: сохранение лучшей модели по validation accuracy
    2. EarlyStopping: остановка при отсутствии улучшений
    3. ReduceLROnPlateau: уменьшение learning rate при стагнации
    
    Returns:
        Список callback-ов
    """
    # Сохранение лучшей модели
    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,  # Сохраняем полную модель (.keras)
        mode='max',
        verbose=1
    )
    
    # Ранняя остановка при отсутствии улучшений
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,  # Ждём 7 эпох без улучшений
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    
    # Уменьшение learning rate при плато
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Уменьшаем в 2 раза
        patience=4,  # Ждём 4 эпохи
        min_lr=1e-7,  # Минимальный порог
        mode='min',
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]


def train():
    """
    Основная функция обучения модели.
    
    Шаги:
    1. Создание генераторов данных
    2. Создание и компиляция модели
    3. Обучение с callbacks
    4. Сохранение финальной модели
    
    Адаптация под 3 класса:
    - Модель создаётся с NUM_CLASSES=3
    - Loss: categorical_crossentropy (требует one-hot метки)
    - Метрики: accuracy, precision, recall, auc
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ ЭМОЦИЙ (3 КЛАССА)")
    print("=" * 60)
    print(f"Эмоции: {EMOTIONS}")
    print(f"Количество классов: {NUM_CLASSES}")
    print()
    
    # =====================================================================
    # ПРОВЕРКА СУЩЕСТВОВАНИЯ ДАННЫХ
    # =====================================================================
    if not os.path.exists(TRAIN_DIR):
        print(f"ОШИБКА: Директория обучения не найдена: {TRAIN_DIR}")
        print("Создайте структуру папок:")
        print("  dataset/train/Happy/")
        print("  dataset/train/Sad/")
        print("  dataset/train/Neutral/")
        print("  dataset/test/Happy/")
        print("  dataset/test/Sad/")
        print("  dataset/test/Neutral/")
        sys.exit(1)
    
    if not os.path.exists(TEST_DIR):
        print(f"ОШИБКА: Директория валидации не найдена: {TEST_DIR}")
        sys.exit(1)
    
    # =====================================================================
    # СОЗДАНИЕ ГЕНЕРАТОРОВ ДАННЫХ
    # =====================================================================
    train_generator, validation_generator = create_data_generators()
    
    # =====================================================================
    # СОЗДАНИЕ МОДЕЛИ
    # =====================================================================
    print("=" * 60)
    print("СОЗДАНИЕ МОДЕЛИ")
    print("=" * 60)
    model = create_model()
    model.summary()
    print()
    
    # Проверка формы выхода модели
    output_shape = model.output_shape[-1]
    if output_shape != NUM_CLASSES:
        print(f"ОШИБКА: Выход модели ({output_shape}) не совпадает с NUM_CLASSES ({NUM_CLASSES})")
        sys.exit(1)
    print(f"✓ Выход модели адаптирован для {NUM_CLASSES} классов")
    print()
    
    # =====================================================================
    # CALLBACKS
    # =====================================================================
    callbacks = create_callbacks()
    
    # =====================================================================
    # ОБУЧЕНИЕ
    # =====================================================================
    print("=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    print(f"Эпох: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Сохранение модели в: {MODEL_PATH}")
    print()
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # =====================================================================
    # ЗАВЕРШЕНИЕ
    # =====================================================================
    print()
    print("=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Модель сохранена: {MODEL_PATH}")
    print()
    
    # Вывод лучших метрик
    if 'val_accuracy' in history.history:
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        print(f"Лучшая точность валидации: {best_val_acc:.4f} (эпоха {best_epoch})")
    
    print()
    print("Для запуска инференса выполните:")
    print("  python src/inference.py")
    print()
    
    return history


if __name__ == "__main__":
    train()
