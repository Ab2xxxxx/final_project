import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def format_classification_report(report):
    report_data = []
    lines = report.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('0') or line.startswith('1'):
            parts = list(filter(None, line.split('  ')))
            if len(parts) >= 5:
                row = {
                    'class': parts[0].strip(),
                    'precision': float(parts[1]),
                    'recall': float(parts[2]),
                    'f1-score': float(parts[3]),
                    'support': int(parts[4])
                }
                report_data.append(row)
                
        elif 'accuracy' in line:
            parts = list(filter(None, line.replace('  ', ' ').split(' ')))
            if len(parts) >= 3:
                report_data.append({
                    'class': 'accuracy',
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1-score': float(parts[1]),
                    'support': int(parts[2])
                })
                
        elif 'macro avg' in line or 'weighted avg' in line:
            parts = list(filter(None, line.split('  ')))
            if len(parts) >= 5:
                row = {
                    'class': parts[0].strip(),
                    'precision': float(parts[1]),
                    'recall': float(parts[2]),
                    'f1-score': float(parts[3]),
                    'support': int(parts[4])
                }
                report_data.append(row)
    
    df = pd.DataFrame(report_data)
    df = df.set_index('class')
    
    return df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}',
        'f1-score': '{:.2f}',
        'support': '{:d}'
    }, na_rep="-").set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    }).set_table_styles([{
        'selector': 'th',
        'props': [('font-size', '14px'), ('text-align', 'left')]
    }])

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               annot_kws={"size": 12}, cbar=False)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    ax.set_title('Confusion Matrix', fontsize=11, pad=10)
    plt.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, auc_score):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', linewidth=1.5)
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='grey')
    ax.set_xlabel('FPR', fontsize=9)
    ax.set_ylabel('TPR', fontsize=9)
    ax.set_title('ROC Curve', fontsize=11, pad=10)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig

def analysis_and_model_page():
    st.title("Прогнозирование отказов оборудования")
    
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
    
        data = data.drop(columns=['Product ID', 'UDI', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        label_e = LabelEncoder()
        data['Type'] = label_e.fit_transform(data['Type'])
        data = data.dropna()

        if 'Machine failure' not in data.columns:
            st.error('Ошибка: В датасете должна быть колонка "Machine failure"')
            return

        X = data.drop('Machine failure', axis=1)
        y = data['Machine failure']
        feature_names = X.columns.tolist()

     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
       
        standard = StandardScaler()
        X_train_st = standard.fit_transform(X_train)
        X_test_st = standard.transform(X_test)

      
        models = {
            "Логистическая регрессия": LogisticRegression(max_iter=1000),
            "Случайный лес": RandomForestClassifier(n_estimators=100, random_state=42),
            "Градиентный бустинг": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "Метод опорных векторов": SVC(probability=True)
        }

        for model_name, model in models.items():
            st.header(model_name)
            

            model.fit(X_train_st, y_train)
            y_pred = model.predict(X_test_st)
            y_proba = model.predict_proba(X_test_st)[:, 1]
            

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            
            col1, col2 = st.columns([1.2, 1], gap="medium")
            
            with col1:
                st.subheader("Метрики качества")
                st.metric("Accuracy", f"{acc:.2%}")
                
                st.subheader("Отчет классификации")
                styled_report = format_classification_report(cr)
                st.dataframe(styled_report, use_container_width=True)
                
            with col2:
                st.subheader("Визуализации")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.pyplot(plot_confusion_matrix(cm), use_container_width=True)
                
                with viz_col2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    st.pyplot(plot_roc_curve(fpr, tpr, roc_auc_score(y_test, y_proba)), 
                            use_container_width=True)
            
            st.markdown("---")

        st.header("Прогнозирование в реальном времени")
        with st.expander("Введите параметры оборудования", expanded=True):
            cols = st.columns(2)
            inputs = {
                "Type": cols[0].selectbox("Тип оборудования", ["L", "M", "H"]),
                "Air temperature [K]": cols[1].number_input("Температура воздуха (K)"),
                "Process temperature [K]": cols[0].number_input("Температура процесса (K)"),
                "Rotational speed [rpm]": cols[1].number_input("Скорость вращения (об/мин)"),
                "Torque [Nm]": cols[0].number_input("Крутящий момент (Нм)"),
                "Tool wear [min]": cols[1].number_input("Износ инструмента (мин)")
            }
            
            if st.button("Выполнить прогноз", type="primary"):
                input_df = pd.DataFrame([inputs]).reindex(columns=feature_names)
                input_df['Type'] = label_e.transform(input_df['Type'])
                input_st = standard.transform(input_df)
                
                best_model_name = max(models, 
                                    key=lambda x: accuracy_score(y_test, models[x].predict(X_test_st)))
                best_model = models[best_model_name]
                prediction = best_model.predict(input_st)[0]
                proba = best_model.predict_proba(input_st)[0][1]
                
                st.success(f"**Рекомендуемая модель:** {best_model_name.split()[-1]}")
                
                if prediction == 1:
                    st.error(f"Вероятность отказа: {proba:.2%}")
                else:
                    st.success(f"Вероятность безотказной работы: {1-proba:.2%}")

if __name__ == "__main__":
    analysis_and_model_page()