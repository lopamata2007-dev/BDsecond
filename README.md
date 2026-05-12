# Распознавание эмоций (3 класса)

Проект распознавания эмоций по изображениям лиц с использованием гибридной архитектуры CNN + BiLSTM + Attention.

Адаптировано для **3 классов эмоций**: Happy, Sad, Neutral.

## 📁 Структура проекта

```
project/
├── src/
│   ├── __init__.py          # Пакет src
│   ├── config.py            # Конфигурация (пути, параметры, список эмоций)
│   ├── model.py             # Создание модели CNN+BiLSTM+Attention
│   ├── train.py             # Обучение модели
│   └── inference.py         # Инференс через веб-камеру
├── dataset/
│   ├── train/               # Обучающие данные
│   │   ├── Happy/
│   │   ├── Sad/
│   │   └── Neutral/
│   └── test/                # Валидационные данные
│       ├── Happy/
│       ├── Sad/
│       └── Neutral/
├── models/
│   └── emotion_model.keras  # Сохранённая модель (после обучения)
├── requirements.txt         # Зависимости
└── README.md               # Этот файл
```

## 📋 Требования

- Python 3.8+
- TensorFlow 2.15+
- OpenCV 4.x
- NumPy

## 🔧 Установка

```bash
pip install tensorflow opencv-python numpy
```

Или из requirements.txt:

```bash
pip install -r requirements.txt
```

## 📊 Подготовка датасета

### Структура папок

Создайте следующую структуру для 3 эмоций:

```
dataset/
├── train/
│   ├── Happy/
│   │   ├── image001.png
│   │   ├── image002.png
│   │   └── ...
│   ├── Sad/
│   │   ├── image001.png
│   │   └── ...
│   └── Neutral/
│       ├── image001.png
│       └── ...
└── test/
    ├── Happy/
    ├── Sad/
    └── Neutral/
```

### Требования к изображениям

- Формат: PNG, JPG
- Размер: любой (будет автоматически приведён к 48x48)
- Цвет: grayscale или RGB (конвертируется автоматически)
- Нормализация: выполняется автоматически (пиксели в [0, 1])

### Рекомендуемые датасеты

1. **FER-2013** — можно отфильтровать только 3 нужные эмоции
2. **CK+** — лабораторный датасет с базовыми эмоциями
3. **Собственный датасет** — соберите изображения через веб-камеру

## 🚀 Запуск

### 1. Обучение модели

```bash
python src/train.py
```

Параметры обучения (в `src/config.py`):
- `EPOCHS = 30`
- `BATCH_SIZE = 32`
- `EMOTIONS = ["Happy", "Sad", "Neutral"]`

Модель сохраняется в `models/emotion_model.keras`.

### 2. Инференс (веб-камера)

```bash
python src/inference.py
```

Клавиши управления:
- **q** — выход
- **r** — сброс сглаживания предсказаний

## 🏗️ Архитектура модели

```
Input (48, 48, 1)
    ↓
CNN Block:
  - Conv2D(32, 3x3) + MaxPool(2,2)
  - Conv2D(64, 3x3) + MaxPool(2,2)
  - Conv2D(128, 3x3) + MaxPool(2,2)
    ↓
Output: (6, 6, 128)
    ↓
Reshape → (36, 128)
    ↓
BiLSTM(64, return_sequences=True)
    ↓
Attention Layer
    ↓
Concatenate([LSTM_out, Attention_out])
    ↓
Flatten → Dense(128, relu) → Dropout(0.5)
    ↓
Dense(3, softmax)  ← АДАПТИРОВАНО ДЛЯ 3 КЛАССОВ!
    ↓
Output: [P(Happy), P(Sad), P(Neutral)]
```

## ⚙️ Оптимизации инференса

1. **Детекция на уменьшенном кадре** (`DETECTION_SCALE=0.5`)
   - Haar Cascade работает на кадре 320x240 вместо 640x480
   - Ускорение в ~4 раза

2. **Предсказание не каждый кадр** (`PREDICT_INTERVAL=5`)
   - Модель запускается каждые 5 кадров
   - Снижение нагрузки на GPU/CPU

3. **Сглаживание предсказаний** (`SMOOTHING_FACTOR=0.3`)
   - Экспоненциальное скользящее среднее
   - Устранение "дрожания" меток

## ✏️ Изменение списка эмоций

Для изменения набора эмоций отредактируйте `src/config.py`:

```python
EMOTIONS = ["Happy", "Sad", "Neutral", "Angry"]  # Добавить 4-ю эмоцию
```

Затем переобучите модель:

```bash
python src/train.py
```

## 📝 Лицензия

MIT License
