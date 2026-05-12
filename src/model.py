"""
Модуль создания гибридной модели CNN + BiLSTM + Attention.

Архитектура адаптирована для 3 классов эмоций (Happy, Sad, Neutral).
Финальный слой: Dense(3, activation="softmax").

Структура модели:
1. CNN блок: 3 слоя Conv2D + MaxPooling для извлечения пространственных признаков
2. Reshape: преобразование 2D карты признаков в последовательность для LSTM
3. BiLSTM: двунаправленный LSTM для анализа временных зависимостей
4. Attention: механизм внимания для выделения важных частей последовательности
5. Классификатор: Dense + Dropout + Softmax для 3 классов
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import (
    TARGET_SIZE,
    IMG_CHANNELS,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    POOL_SIZE,
    CNN_OUTPUT_RESHAPED,
    LSTM_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
    NUM_CLASSES,
    OPTIMIZER,
    LOSS_FUNCTION,
    METRICS,
)


def create_cnn_block(inputs):
    """
    Создаёт CNN блок из 3 свёрточных слоёв с MaxPooling.
    
    Архитектура:
    - Conv2D(32, 3x3, relu) + MaxPooling(2,2)
    - Conv2D(64, 3x3, relu) + MaxPooling(2,2)
    - Conv2D(128, 3x3, relu) + MaxPooling(2,2)
    
    Вход: (48, 48, 1)
    Выход: (6, 6, 128) — после 3 уменьшений в 2 раза: 48->24->12->6
    
    Args:
        inputs: Входной тензор Keras
        
    Returns:
        Выходной тензор CNN блока
    """
    x = inputs
    
    # Блок 1: 32 фильтра
    x = layers.Conv2D(CNN_FILTERS[0], CNN_KERNEL_SIZE, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(POOL_SIZE)(x)
    
    # Блок 2: 64 фильтра
    x = layers.Conv2D(CNN_FILTERS[1], CNN_KERNEL_SIZE, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(POOL_SIZE)(x)
    
    # Блок 3: 128 фильтров
    x = layers.Conv2D(CNN_FILTERS[2], CNN_KERNEL_SIZE, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(POOL_SIZE)(x)
    
    return x


def create_attention_layer(recurrent_output):
    """
    Создаёт слой Attention для выделения важных частей последовательности.
    
    Механизм внимания:
    - Вычисляет веса важности для каждого временного шага LSTM
    - Взвешивает выходы LSTM согласно этим весам
    - Позволяет модели фокусироваться на наиболее информативных признаках
    
    Args:
        recurrent_output: Выход BiLSTM слоя, форма (batch, timesteps, lstm_units*2)
        
    Returns:
        Context vector после применения attention, форма (batch, lstm_units*2)
    """
    # Self-attention: attention([query, value]) где query=value=recurrent_output
    attention_output = layers.Attention()([recurrent_output, recurrent_output])
    return attention_output


def create_model():
    """
    Создаёт и компилирует гибридную модель CNN + BiLSTM + Attention.
    
    Адаптация под 3 класса:
    - Финальный слой Dense изменён с Dense(7) на Dense(3)
    - Activation: softmax для многоклассовой классификации
    - Loss: categorical_crossentropy (требует one-hot encoded меток)
    
    Полная архитектура:
    1. Input: (48, 48, 1) — grayscale изображение
    2. CNN: 3 блока Conv2D+MaxPool → выход (6, 6, 128)
    3. Reshape: (6, 6, 128) → (36, 128) — последовательность из 36 векторов
    4. BiLSTM: Bidirectional(LSTM(64, return_sequences=True)) → (batch, 36, 128)
       * Двунаправленный LSTM обрабатывает последовательность в обоих направлениях
       * return_sequences=True возвращает выход для каждого временного шага
    5. Attention: взвешивание выходов LSTM → (batch, 128)
    6. Concatenate: объединение attention output с original recurrent output
    7. Flatten + Dense(128, relu) + Dropout(0.5)
    8. Output: Dense(3, softmax) — вероятности для 3 эмоций
    
    Returns:
        Скомпилированная модель Keras
    """
    # =====================================================================
    # ВХОДНОЙ СЛОЙ
    # =====================================================================
    # Вход: 48x48 grayscale изображение (1 канал)
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # =====================================================================
    # CNN БЛОК: Извлечение пространственных признаков
    # =====================================================================
    # CNN обрабатывает изображение как 2D структуру, выявляя локальные паттерны:
    # - Края, текстуры, формы facial features
    cnn_output = create_cnn_block(inputs)
    # Выход CNN: (batch_size, 6, 6, 128)
    
    # =====================================================================
    # RESHAPE: Подготовка данных для LSTM
    # =====================================================================
    # Преобразуем 2D карту признаков (6, 6, 128) в последовательность (36, 128)
    # Это позволяет трактовать пространственные позиции как "временные шаги"
    reshaped = layers.Reshape(CNN_OUTPUT_RESHAPED)(cnn_output)
    # Выход: (batch_size, 36, 128) — 36 "временных шагов" по 128 признаков
    
    # =====================================================================
    # BiLSTM БЛОК: Анализ контекстных зависимостей
    # =====================================================================
    # Двунаправленный LSTM обрабатывает последовательность в двух направлениях:
    # - Forward LSTM: слева направо
    # - Backward LSTM: справа налево
    # Объединение выходов даёт полный контекст для каждой позиции
    recurrent = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True)
    )(reshaped)
    # Выход: (batch_size, 36, 128) — 64*2 = 128 единиц (forward + backward)
    
    # =====================================================================
    # ATTENTION БЛОК: Выделение важных признаков
    # =====================================================================
    # Механизм внимания вычисляет веса важности для каждого временного шага
    # и создаёт контекстный вектор, фокусируясь на наиболее релевантных участках
    attention_output = create_attention_layer(recurrent)
    # Выход: (batch_size, 36, 128) — attention-weighted выходы
    
    # =====================================================================
    # CONCATENATE: Объединение attention и original outputs
    # =====================================================================
    # Конкатенация позволяет сохранить как исходные признаки LSTM,
    # так и обогащённые attention векторы
    combined = layers.Concatenate()([recurrent, attention_output])
    # Выход: (batch_size, 36, 256) — 128 + 128
    
    # =====================================================================
    # КЛАССИФИКАТОР
    # =====================================================================
    # Flatten: преобразование последовательности в плоский вектор
    flattened = layers.Flatten()(combined)
    # Выход: (batch_size, 36*256) = (batch_size, 9216)
    
    # Dense слой с ReLU активацией
    dense = layers.Dense(DENSE_UNITS, activation='relu')(flattened)
    # Выход: (batch_size, 128)
    
    # Dropout для регуляризации (предотвращение переобучения)
    dropout = layers.Dropout(DROPOUT_RATE)(dense)
    
    # =====================================================================
    # ФИНАЛЬНЫЙ СЛОЙ: Адаптировано для 3 классов!
    # =====================================================================
    # Изменено с Dense(7) на Dense(3) для работы с 3 эмоциями
    # Softmax преобразует logits в вероятности (сумма = 1)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(dropout)
    # Выход: (batch_size, 3) — вероятности [P(Happy), P(Sad), P(Neutral)]
    
    # =====================================================================
    # СОЗДАНИЕ И КОМПИЛЯЦИЯ МОДЕЛИ
    # =====================================================================
    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_Attention_3Emotions")
    
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )
    
    return model


# Константы для импорта в других модулях
IMG_HEIGHT = TARGET_SIZE[0]
IMG_WIDTH = TARGET_SIZE[1]
