import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time

def evaluate_model(model, model_name):

    '''Функция для обучения и оценки классических ML моделей (XGBoost / LightGBM)

    На вход принимает:
    model – обучаемая модель
    model_name – строка с названием модели для логирования

    Внутри выполняется:
    - обучение модели на тренировочной выборке (X_cord_train, y_train)
    - замер времени инференса на тестовой выборке
    - расчёт accuracy
    - построение classification report
    - построение confusion matrix

    На выходе функция ничего не возвращает, но печатает:
    - время инференса модели
    - точность (accuracy)
    - classification report по классам
    - confusion matrix в виде heatmap'''


    model.fit(X_cord_train, y_train)

    start = time.perf_counter()
    y_pred = model.predict(X_cords_test)
    end = time.perf_counter()

    time_to_test = round((end - start),4)

    print(f"Время на тест: {time_to_test}")

    accuracy = round((accuracy_score(y_test, y_pred)),4)
    print(f"Точность: {model_name}: {accuracy}")

    print("Classification Report")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title(f'Матрица ошибок для {model_name}')
    plt.show()

'''Запуск обучения модели XGBoost'''
print("XGBoost Classification Report")
XGB_model = xgb.XGBClassifier(n_estimators=50,
                              learning_rate=0.1,
                              max_depth=3,
                              random_state=42)

evaluate_model(XGB_model, model_name="XGBoost")

'''Запуск обучения модели LightGBM'''
print("LightGBM Classification Report")
lightgbm_model = lgb.LGBMClassifier(n_estimators=50,
                                    learning_rate=0.1,
                                    max_depth=3,
                                    random_state=42)

evaluate_model(lightgbm_model, model_name="LightGBM")