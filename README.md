# README.md

# ВКР: Бинарная классификация для предиктивного обслуживания оборудования

## Описание
Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (`Machine failure = 1`) или нет (`Machine failure = 0`).  
Проект реализован в виде Streamlit-приложения с поддержкой загрузки собственных данных и слайд-презентацией.

---

## Структура проекта

```
FinalWork/
├── app.py                  # Главный файл Streamlit-приложения
├── analysis_and_model.py   # Основная страница с анализом и моделями
├── presentation.py         # Страница с презентацией проекта
├── requirements.txt        # Зависимости
├── data/
│   └── predictive_maintenance.csv
└── README.md               # Этот файл
```

---

## Данные

Используется датасет **AI4I 2020 Predictive Maintenance Dataset**, доступный на UCI:
https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset

Каждая запись описывает параметры работы оборудования.  
Признаки включают температуру, скорость, износ и т.д.  
Целевая переменная — `Machine failure` (0 или 1).

---

## Как запустить проект

### 1. Клонируйте репозиторий

```bash
git clone
cd final_project
```

### 2. Установите зависимости

```bash
pip install -r requirements.txt
```

### 3. Запустите приложение

```bash
streamlit run app.py
```

---

## Возможности приложения

- Загрузка собственного CSV-файла
- Обработка и масштабирование признаков
- Обучение моделей: Logistic Regression, Random Forest, XGBoost
- Визуализация:
  - Accuracy
  - Confusion Matrix
  - ROC-AUC
- Презентация проекта с помощью `streamlit-reveal-slides`

---

## Видео-демонстрация
[Ссылка на видео](video/video.mp4)

---