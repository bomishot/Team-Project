import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# RandomForestClassifier 가 가장 높은 성능을 보임

def train_random_forest_classifier(df):  
    # target 데이터 LabelEncoder
    X = df.drop("targets", axis=1)
    y = df['targets']

    # 데이터를 훈련용(train)과 테스트용(test)으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomForestClassifier 모델 생성과 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train,  y_train)

    # 모델 평가 및 결과 출력
    model_pred = model.predict(X_test)
    report = classification_report(y_test, model_pred)
    accuracy = round(model.score(X_test, y_test) * 100, 1)

    print("Classification Report:")
    print(report)
    print(f'RandomForestClassifier: class 조절 정확도 (accuracy) {accuracy}%')
