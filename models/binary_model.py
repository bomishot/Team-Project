from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def binary_model(df):
    df_origin = df.copy()

    X = df.drop(columns='targets')
    Y = df['targets']

    # randomoversampling
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, Y)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['targets'] = y_resampled
    df = df_resampled

    # X, y 분리
    X_origin = df_origin.drop(columns='targets')
    y_origin = df_origin['targets']

    # Train, Test Dataset : Test는 원본데이터로.
    X_origin_train, X_origin_test, y_origin_train, y_origin_test = train_test_split(X_origin, y_origin, test_size=0.2, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Modeling - XGBoost Classifier
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Classification Report
    model_pred = model.predict(X_origin_test)
    report = classification_report(y_origin_test, model_pred)
    accuracy = round(model.score(X_origin_test, y_origin_test) * 100, 1)

    print("binary Report:")
    print(report)
    print(f'BinaryClassifier: class 조절 정확도 (accuracy) {accuracy}%')

    return binary_model