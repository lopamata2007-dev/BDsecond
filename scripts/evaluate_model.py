"""
Скрипт оценки модели и визуализации результатов.

Использование:
    python scripts/evaluate_model.py

Функционал:
1. Загрузка обученной модели
2. Оценка на тестовых данных
3. Построение матрицы ошибок (confusion matrix)
4. Построение графиков истории обучения
5. Отчёт с интерпретацией результатов
"""

import os
import sys
import json
import numpy as np

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

from src.config import (
    MODEL_PATH,
    TEST_DIR,
    TARGET_SIZE,
    COLOR_MODE,
    BATCH_SIZE,
    EMOTIONS,
    NUM_CLASSES,
    REPORTS_DIR,
)


def load_trained_model():
    """Загружает обученную модель."""
    if not os.path.exists(MODEL_PATH):
        print(f"ОШИБКА: Модель не найдена: {MODEL_PATH}")
        print("Сначала запустите обучение: python src/train.py")
        sys.exit(1)
    
    print(f"Загрузка модели из: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"✓ Модель загружена")
    return model


def create_test_generator():
    """Создаёт генератор для тестовых данных."""
    if not os.path.exists(TEST_DIR):
        print(f"ОШИБКА: Тестовая директория не найдена: {TEST_DIR}")
        sys.exit(1)
    
    datagen = ImageDataGenerator(rescale=1./255)
    
    generator = datagen.flow_from_directory(
        directory=TEST_DIR,
        target_size=TARGET_SIZE,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False  # Важно для корректной оценки
    )
    
    print(f"✓ Загружено {generator.samples} тестовых изображений")
    return generator


def evaluate_model(model, generator):
    """
    Оценивает модель на тестовых данных.
    
    Returns:
        results: Словарь с метриками
        y_true: Истинные метки
        y_pred: Предсказанные метки
        y_probs: Вероятности предсказаний
    """
    print("\n" + "=" * 60)
    print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 60)
    
    # Оценка модели
    results = model.evaluate(generator, verbose=1)
    metrics_names = ['loss'] + [m.name for m in model.metrics]
    
    print("\n" + "-" * 60)
    print("МЕТРИКИ:")
    for name, value in zip(metrics_names, results):
        print(f"  {name}: {value:.4f}")
    
    # Получение предсказаний
    print("\nВычисление предсказаний...")
    y_probs = model.predict(generator)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = generator.classes
    
    # Точность
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nОбщая точность (accuracy): {accuracy:.4f}")
    
    return {
        'metrics': dict(zip(metrics_names, results)),
        'accuracy': accuracy
    }, y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Строит и сохраняет матрицу ошибок.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        save_path: Путь для сохранения графика
    """
    print("\nПостроение матрицы ошибок...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix - Emotion Recognition (3 Classes)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Матрица ошибок сохранена: {save_path}")
    
    plt.show()
    
    return cm


def print_classification_report(y_true, y_pred):
    """Выводит подробный отчёт о классификации."""
    print("\n" + "=" * 60)
    print("ПОДРОБНЫЙ ОТЧЁТ О КЛАССИФИКАЦИИ")
    print("=" * 60)
    
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTIONS,
        digits=4
    )
    print(report)
    
    return report


def plot_training_history(history_path=None):
    """
    Строит графики истории обучения.
    
    Args:
        history_path: Путь к JSON файлу с историей (если есть)
    """
    # Проверяем наличие файла истории
    history_file = os.path.join(REPORTS_DIR, "training_history.json")
    
    if history_path and os.path.exists(history_path):
        history_file = history_path
    
    if not os.path.exists(history_file):
        print("\nФайл истории обучения не найден. Пропускаем построение графиков.")
        return
    
    print("\nЗагрузка истории обучения...")
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train Loss')
    axes[0, 1].plot(history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision & Recall (если есть)
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Train Precision')
        axes[1, 0].plot(history.get('val_precision', history['precision']), label='Val Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Train Recall')
        axes[1, 1].plot(history.get('val_recall', history['recall']), label='Val Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(REPORTS_DIR, "metrics_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Графики метрик сохранены: {save_path}")
    
    plt.show()


def interpret_results(results, cm, report):
    """
    Генерирует интерпретацию результатов.
    
    Args:
        results: Словарь с метриками
        cm: Матрица ошибок
        report: Отчёт о классификации
    """
    print("\n" + "=" * 60)
    print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    accuracy = results['accuracy']
    
    print(f"""
1. ОБЩАЯ ТОЧНОСТЬ: {accuracy:.2%}
   {'✓ Отличный результат!' if accuracy >= 0.85 else 
     '✓ Хороший результат' if accuracy >= 0.70 else 
     '⚠ Требуется улучшение модели'}

2. АНАЛИЗ ПО ЭМОЦИЯМ:
""")
    
    # Анализ по каждой эмоции
    for i, emotion in enumerate(EMOTIONS):
        tp = cm[i, i]
        total_actual = cm[i, :].sum()
        total_predicted = cm[:, i].sum()
        
        recall = tp / total_actual if total_actual > 0 else 0
        precision = tp / total_predicted if total_predicted > 0 else 0
        
        print(f"   {emotion}:")
        print(f"      - Правильно распознано: {tp}/{total_actual} ({recall:.1%})")
        print(f"      - Precision: {precision:.1%}")
        
        if recall < 0.7:
            print(f"      ⚠ Низкий recall — модель плохо распознаёт эту эмоцию")
    
    print(f"""
3. РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:
   - Добавьте больше обучающих данных для эмоций с низким recall
   - Попробуйте увеличить количество эпох обучения
   - Экспериментируйте с параметрами аугментации данных
   - Рассмотрите возможность использования предобученных моделей

4. ПРИМЕНИМОСТЬ МОДЕЛИ:
   {'✓ Модель готова к практическому использованию' if accuracy >= 0.80 else 
     '⚠ Модель требует доработки для production' if accuracy >= 0.65 else 
     '❌ Модель нуждается в значительном улучшении'}
""")


def main():
    """Основная функция оценки модели."""
    print("=" * 60)
    print("ОЦЕНКА МОДЕЛИ РАСПОЗНАВАНИЯ ЭМОЦИЙ (3 КЛАССА)")
    print("=" * 60)
    print(f"Эмоции: {EMOTIONS}")
    print(f"Количество классов: {NUM_CLASSES}")
    
    # Загрузка модели
    model = load_trained_model()
    
    # Создание тестового генератора
    generator = create_test_generator()
    
    # Оценка модели
    results, y_true, y_pred, y_probs = evaluate_model(model, generator)
    
    # Построение матрицы ошибок
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    cm = plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    # Подробный отчёт
    report = print_classification_report(y_true, y_pred)
    
    # Графики истории обучения
    plot_training_history()
    
    # Интерпретация результатов
    interpret_results(results, cm, report)
    
    # Сохранение результатов в JSON
    results_file = os.path.join(REPORTS_DIR, "evaluation_results.json")
    eval_data = {
        'metrics': results['metrics'],
        'accuracy': results['accuracy'],
        'emotions': EMOTIONS,
        'num_classes': NUM_CLASSES,
        'confusion_matrix': cm.tolist(),
        'samples_evaluated': len(y_true)
    }
    
    with open(results_file, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"\n✓ Результаты сохранены: {results_file}")
    print("\n" + "=" * 60)
    print("ОЦЕНКА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()
