"""
Скрипт инференса (распознавание эмоций в реальном времени через веб-камеру).

Реализованы оптимизации из технического отчёта:
1. Детекция лиц на уменьшенном кадре (DETECTION_SCALE=0.5)
2. Предсказание модели не на каждом кадре, а каждые N кадров (PREDICT_INTERVAL=5)
3. Плавный вывод с использованием сглаживания предсказаний (SMOOTHING_FACTOR=0.3)

Адаптация под 3 класса:
- Модель ожидает 3 выхода (Happy, Sad, Neutral)
- Отрисовка цветовых меток для 3 эмоций
- Проверки и константы обновлены для 3 классов
"""

import os
import sys
import time
from collections import deque

# Добавляем корень проекта в path для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src.config import (
    MODEL_PATH,
    EMOTIONS,
    NUM_CLASSES,
    IDX_TO_EMOTION,
    TARGET_SIZE,
    DETECTION_SCALE,
    PREDICT_INTERVAL,
    SMOOTHING_FACTOR,
    FACE_CASCADE_PATH,
    MIN_FACE_SIZE,
    EMOTION_COLORS,
)


def load_model():
    """
    Загружает обученную модель Keras.
    
    Returns:
        model: Загруженная модель TensorFlow/Keras
    """
    from tensorflow.keras.models import load_model
    
    if not os.path.exists(MODEL_PATH):
        print(f"ОШИБКА: Модель не найдена по пути: {MODEL_PATH}")
        print("Сначала запустите обучение: python src/train.py")
        sys.exit(1)
    
    print(f"Загрузка модели из: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"✓ Модель загружена успешно")
    print(f"  Вход: {model.input_shape}")
    print(f"  Выход: {model.output_shape} ({NUM_CLASSES} классов)")
    
    return model


def load_face_cascade():
    """
    Загружает Haar Cascade классификатор для детекции лиц.
    
    Returns:
        face_cascade: OpenCV CascadeClassifier
    """
    if FACE_CASCADE_PATH is None:
        print("ОШИБКА: Не удалось загрузить Haar Cascade")
        print("Проверьте установку OpenCV: pip install opencv-python")
        sys.exit(1)
    
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    if face_cascade.empty():
        print(f"ОШИБКА: Не удалось загрузить каскад из {FACE_CASCADE_PATH}")
        sys.exit(1)
    
    return face_cascade


def detect_faces(frame, face_cascade):
    """
    Обнаруживает лица на кадре с использованием Haar Cascade.
    
    Оптимизация: Детекция выполняется на уменьшенной копии кадра
    (DETECTION_SCALE=0.5), что ускоряет обработку в ~4 раза.
    
    Args:
        frame: Исходный кадр (BGR)
        face_cascade: Загруженный Haar Cascade классификатор
        
    Returns:
        faces: Массив координат лиц (x, y, w, h)
        small_gray: Уменьшенная grayscale копия (для отладки)
    """
    # Создаём уменьшенную копию для детекции
    small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    
    # Конвертируем в grayscale (Haar Cascade требует одноканальное изображение)
    small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Применяем эквализацию гистограммы для улучшения контраста
    small_gray = cv2.equalizeHist(small_gray)
    
    # Детекция лиц на уменьшенном изображении
    # scaleFactor=1.1, minNeighbors=5 — стандартные параметры
    faces_small = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    
    # Масштабируем координаты обратно к исходному размеру кадра
    faces = []
    for (x, y, w, h) in faces_small:
        faces.append((
            int(x / DETECTION_SCALE),
            int(y / DETECTION_SCALE),
            int(w / DETECTION_SCALE),
            int(h / DETECTION_SCALE)
        ))
    
    return faces, small_gray


def preprocess_face(face_image):
    """
    Предобрабатывает изображение лица для подачи в модель.
    
    Шаги предобработки:
    1. Конвертация в grayscale
    2. Изменение размера до 48x48
    3. Нормализация пикселей в [0, 1]
    4. Добавление размерности батча (1, 48, 48, 1)
    
    Args:
        face_image: Вырезанное изображение лица (BGR)
        
    Returns:
        processed: Готовый тензор для модели (1, 48, 48, 1)
    """
    # Конвертация в grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Изменение размера до 48x48 (вход модели)
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    # Нормализация в [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Добавление размерности батча: (48, 48) -> (1, 48, 48, 1)
    processed = np.expand_dims(normalized, axis=(0, -1))
    
    return processed


def predict_emotion(model, face_image, previous_probs=None):
    """
    Предсказывает эмоцию по изображению лица.
    
    Args:
        model: Обученная модель
        face_image: Изображение лица
        previous_probs: Предыдущие вероятности для сглаживания (или None)
        
    Returns:
        emotion: Название эмоции (строка)
        probs: Вектор вероятностей (3,)
        confidence: Уверенность модели (максимальная вероятность)
    """
    # Предобработка
    processed = preprocess_face(face_image)
    
    # Предсказание модели
    probs = model.predict(processed, verbose=0)[0]
    
    # Сглаживание предсказаний (temporal smoothing)
    # Если есть предыдущие вероятности, используем экспоненциальное сглаживание
    if previous_probs is not None:
        probs = SMOOTHING_FACTOR * probs + (1 - SMOOTHING_FACTOR) * previous_probs
    
    # Определение эмоции с максимальной вероятностью
    emotion_idx = np.argmax(probs)
    emotion = IDX_TO_EMOTION[emotion_idx]
    confidence = probs[emotion_idx]
    
    return emotion, probs, confidence


class EmotionSmoother:
    """
    Класс для сглаживания предсказаний эмоций во времени.
    
    Использует скользящее среднее по последним N предсказаниям
    для устранения "дрожания" меток при видеопотоке.
    """
    
    def __init__(self, window_size=5):
        """
        Args:
            window_size: Размер окна для скользящего среднего
        """
        self.window_size = window_size
        self.prob_history = deque(maxlen=window_size)
        self.emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    
    def update(self, probs):
        """
        Обновляет историю предсказаний и возвращает сглаженную эмоцию.
        
        Args:
            probs: Текущие вероятности от модели (3,)
            
        Returns:
            smoothed_emotion: Сглаженная метка эмоции
            smoothed_probs: Сглаженные вероятности
        """
        # Добавляем текущие вероятности в историю
        self.prob_history.append(probs.copy())
        
        # Вычисляем средние вероятности по истории
        if len(self.prob_history) > 0:
            avg_probs = np.mean(self.prob_history, axis=0)
        else:
            avg_probs = probs
        
        # Определяем эмоцию по средним вероятностям
        emotion_idx = np.argmax(avg_probs)
        smoothed_emotion = IDX_TO_EMOTION[emotion_idx]
        
        return smoothed_emotion, avg_probs
    
    def reset(self):
        """Сбрасывает историю предсказаний."""
        self.prob_history.clear()
        self.emotion_counts = {emotion: 0 for emotion in EMOTIONS}


def draw_emotion_label(frame, x, y, w, h, emotion, confidence, probs):
    """
    Отрисовывает рамку вокруг лица и метку эмоции.
    
    Визуальный отклик включает:
    - Цветная рамка вокруг лица (цвет зависит от эмоции)
    - Текст с названием эмоции
    - Процент уверенности модели
    - Бар с вероятностями для всех 3 эмоций
    
    Args:
        frame: Исходный кадр
        x, y, w, h: Координаты и размеры области лица
        emotion: Предсказанная эмоция
        confidence: Уверенность модели
        probs: Вектор вероятностей для всех эмоций
    """
    # Цвет рамки в зависимости от эмоции
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))  # Белый по умолчанию
    
    # Рисуем рамку вокруг лица
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Подготовка текста
    label = f"{emotion}: {confidence:.1%}"
    
    # Вычисляем размер текста для фона
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Рисуем фон под текстом
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width, y),
        color,
        cv2.FILLED
    )
    
    # Рисуем текст эмоции
    cv2.putText(
        frame,
        label,
        (x, y - 5),
        font,
        font_scale,
        (0, 0, 0),  # Чёрный текст на цветном фоне
        thickness
    )
    
    # Рисуем бар с вероятностями (ниже рамки)
    bar_width = w
    bar_height = 6
    bar_y = y + h + 5
    
    # Фон бара
    cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (200, 200, 200), cv2.FILLED)
    
    # Заполняем бар пропорционально вероятностям
    current_x = x
    for i, (emotion_name, prob) in enumerate(zip(EMOTIONS, probs)):
        segment_width = int(bar_width * prob)
        emotion_color = EMOTION_COLORS.get(emotion_name, (255, 255, 255))
        
        if segment_width > 0:
            cv2.rectangle(
                frame,
                (current_x, bar_y),
                (current_x + segment_width, bar_y + bar_height),
                emotion_color,
                cv2.FILLED
            )
            current_x += segment_width
    
    # Подписываем вероятности
    prob_text = " | ".join([f"{e[0]}:{p:.0f}" for e, p in zip(EMOTIONS, probs)])
    cv2.putText(
        frame,
        prob_text,
        (x, bar_y + bar_height + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )


def run_inference():
    """
    Основной цикл инференса: захват видео, детекция, предсказание, отрисовка.
    
    Оптимизации:
    1. Детекция на downscale-кадре (fx=0.5)
    2. Предсказание каждые PREDICT_INTERVAL кадров
    3. Сглаживание предсказаний через EmotionSmoother
    """
    print("=" * 60)
    print("ИНФЕРЕНС: РАСПОЗНАВАНИЕ ЭМОЦИЙ (3 КЛАССА)")
    print("=" * 60)
    print(f"Эмоции: {EMOTIONS}")
    print(f"Количество классов: {NUM_CLASSES}")
    print()
    
    # Загрузка модели
    model = load_model()
    print()
    
    # Загрузка детектора лиц
    face_cascade = load_face_cascade()
    print(f"✓ Haar Cascade загружен")
    print()
    
    # Инициализация веб-камеры
    print("Инициализация веб-камеры...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ОШИБКА: Не удалось открыть веб-камеру")
        print("Проверьте подключение камеры и права доступа")
        sys.exit(1)
    
    # Установка разрешения камеры (опционально)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✓ Камера инициализирована")
    print()
    
    # Переменные для оптимизации
    frame_count = 0
    current_emotion = "Neutral"  # Эмоция по умолчанию
    current_probs = np.ones(NUM_CLASSES) / NUM_CLASSES  # Равномерное распределение
    current_confidence = 0.0
    last_faces = []  # Последние обнаруженные лица
    
    # Сглаживатель предсказаний
    smoother = EmotionSmoother(window_size=5)
    
    print("Нажмите 'q' для выхода")
    print("-" * 60)
    
    # Статистика FPS
    fps_start_time = time.time()
    fps_frame_count = 0
    
    # =====================================================================
    # ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ КАДРОВ
    # =====================================================================
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ОШИБКА: Не удалось захватить кадр")
            break
        
        frame_count += 1
        fps_frame_count += 1
        
        # =================================================================
        # ДЕТЕКЦИЯ ЛИЦ
        # =================================================================
        # Детекция выполняется на каждом кадре (быстрая операция на уменьшенном кадре)
        faces, _ = detect_faces(frame, face_cascade)
        last_faces = faces
        
        # =================================================================
        # ПРЕДСКАЗАНИЕ ЭМОЦИИ
        # =================================================================
        # Предсказание делается только каждые PREDICT_INTERVAL кадров
        # Это снижает нагрузку на GPU/CPU и делает вывод более плавным
        if frame_count % PREDICT_INTERVAL == 0 and len(faces) > 0:
            # Берём первое обнаруженное лицо (можно расширить для нескольких лиц)
            x, y, w, h = faces[0]
            
            # Вырезаем область лица с небольшим отступом
            face_x = max(0, x - 10)
            face_y = max(0, y - 10)
            face_w = min(frame.shape[1] - face_x, w + 20)
            face_h = min(frame.shape[0] - face_y, h + 20)
            
            face_image = frame[face_y:face_y + face_h, face_x:face_x + face_w]
            
            # Предсказание с сглаживанием
            _, probs, confidence = predict_emotion(model, face_image, current_probs)
            
            # Обновление сглаженных значений
            smoothed_emotion, smoothed_probs = smoother.update(probs)
            
            current_emotion = smoothed_emotion
            current_probs = smoothed_probs
            current_confidence = smoothed_probs[np.argmax(smoothed_probs)]
        
        # =================================================================
        # ОТРИСОВКА РЕЗУЛЬТАТОВ
        # =================================================================
        # Если лица обнаружены — рисуем рамку и метку
        if len(faces) > 0:
            x, y, w, h = faces[0]
            draw_emotion_label(frame, x, y, w, h, current_emotion, current_confidence, current_probs)
        else:
            # Если лиц нет — информируем пользователя
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        # =================================================================
        # ОТРИСОВКА ИНФОРМАЦИИ О FPS И ПАРАМЕТРАХ
        # =================================================================
        # Вычисление FPS каждые 30 кадров
        if fps_frame_count >= 30:
            fps_end_time = time.time()
            fps = fps_frame_count / (fps_end_time - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
        else:
            fps = 0
        
        # Информация в углу кадра
        info_text = [
            f"FPS: {fps:.1f}",
            f"Emotions: {NUM_CLASSES}",
            f"Predict interval: {PREDICT_INTERVAL}",
            f"Detection scale: {DETECTION_SCALE}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(
                frame,
                text,
                (10, frame.shape[0] - 10 - i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        
        # =================================================================
        # ВЫВОД КАДРА
        # =================================================================
        cv2.imshow("Emotion Recognition (3 Classes)", frame)
        
        # Обработка нажатий клавиш
        key = cv2.waitKey(1) & 0xFF
        
        # Выход по 'q'
        if key == ord('q'):
            break
        
        # Сброс сглаживания по 'r'
        if key == ord('r'):
            smoother.reset()
            current_emotion = "Neutral"
            current_probs = np.ones(NUM_CLASSES) / NUM_CLASSES
            print("Сглаживание сброшено")
    
    # =====================================================================
    # ЗАВЕРШЕНИЕ РАБОТЫ
    # =====================================================================
    print()
    print("Завершение работы...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("✓ Веб-камера освобождена")
    print("✓ Окна закрыты")
    print()
    print("Спасибо за использование!")


if __name__ == "__main__":
    run_inference()
