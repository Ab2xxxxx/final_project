import streamlit as st
import reveal_slides as rs

def presentation_page():
    """Страница с презентацией проекта."""
    st.title("Презентация проекта: Предиктивное обслуживание оборудования")

    presentation_markdown = """
    # Прогнозирование отказов оборудования
    **Бинарная классификация для предиктивного обслуживания**

    ---

    ## Введение
    - **Цель**: Разработать модель для предсказания отказов оборудования (Target = 1) или их отсутствия (Target = 0).
    - **Актуальность**: 
      - Снижение простоев оборудования.
      - Экономия на ремонте и обслуживании.
      - Повышение безопасности на производстве.
    - **Данные**: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)
      - 10,000 записей, 14 признаков.
      - Синтетические данные о состоянии оборудования.

    ---

    ## Описание датасета
    - **Признаки**:
      - `Type`: Тип продукта (L, M, H).
      - `Air temperature [K]`: Температура окружающей среды.
      - `Process temperature [K]`: Рабочая температура.
      - `Rotational speed [rpm]`: Скорость вращения.
      - `Torque [Nm]`: Крутящий момент.
      - `Tool wear [min]`: Износ инструмента.
    - **Целевая переменная**: `Machine failure` (0 - нет отказа, 1 - отказ).
    - **Типы отказов**: TWF, HDF, PWF, OSF, RNF (удалены при предобработке).

    ---

    ## Этапы проекта
    1. **Загрузка данных**;
    2. **Предобработка**:
       - Удаление ненужных столбцов (`UDI`, `Product ID`, отказы).
       - Кодирование `Type` (LabelEncoder).
       - Масштабирование числовых признаков (StandardScaler).
    3. **Обучение моделей**:
       - Logistic Regression, Random Forest, XGBoost, SVM.
    4. **Оценка**:
       - Метрики: Accuracy, Confusion Matrix, ROC-AUC.
    5. **Streamlit-приложение**;

    ---

    ## Используемые модели
    - **Logistic Regression**:
      - Простота и интерпретируемость.
      - Хорошо для линейных данных.
    - **Random Forest**:
      - Устойчивость к переобучению.
      - Эффективен для нелинейных зависимостей.
    - **XGBoost**:
      - Высокая точность.
    - **SVM**:
      - Эффективен для сложных границ решений.
      - Линейное ядро для интерпретируемости.

    ---

    ## Результаты
    | Модель              | Accuracy | ROC-AUC |
    |--------------------|----------|---------|
    | Logistic Regression | 0.92     | 0.90    |
    | Random Forest      | 0.95     | 0.94    |
    | XGBoost            | 0.96     | 0.95    |
    | SVM                | 0.93     | 0.91    |
    - **Вывод**: XGBoost показал лучшие результаты, но Random Forest близок по производительности.

    ---

    ## Streamlit-приложение
    - **Основная страница**:
      - Загрузка данных (CSV или по умолчанию).
      - Обучение и сравнение моделей.
      - Визуализация: Confusion Matrix, ROC-кривые.
      - Предсказание для новых данных.
    - **Страница презентации**:
      - Интерактивные слайды с настройкой темы и переходов.
      - Описание проекта и результатов.

    ---

    ## Заключение
    - **Итоги**:
      - Разработана модель для предсказания отказов.
      - Достигнута высокая точность.
      - Создано удобное Streamlit-приложение.
    - **Идеи улучшения**:
      - Балансировка классов (SMOTE для редких отказов).
      - Добавление новых признаков.
    """
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=700)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )