from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd


def multi_classification(df):  
    # target 데이터 LabelEncoder
    X = df.drop("targets", axis=1)
    y = df['targets']

    # 데이터를 훈련용(train)과 테스트용(test)으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomForestClassifier 모델 생성
    model = RandomForestClassifier(n_estimators=300, 
                                   max_depth=30,
                                   max_features='sqrt',
                                   min_samples_leaf=1,
                                   min_samples_split=5,
                                   bootstrap=False,
                                   class_weight='balanced',
                                   random_state=42)

    # 모델 학습
    model.fit(X_train, y_train)

    # Feature Importance 확인
    feature_importance = model.feature_importances_
    feature_names = X_train.columns

    # 중요도가 낮은 피처 제거
    threshold = 0.05  # 임계값 (임의로 설정, 조정 가능)
    selected_features = feature_names[feature_importance <= threshold]

    X_train_selected = X_train.drop(selected_features, axis=1)
    X_test_selected = X_test.drop(selected_features, axis=1)

    # 모델 다시 학습
    model.fit(X_train_selected, y_train)

    # 모델 평가 및 결과 출력
    model_pred = model.predict(X_test_selected)
    report = classification_report(y_test, model_pred)
    accuracy = round(model.score(X_test_selected, y_test) * 100, 2)

    print("Classification Report:")
    print(report)
    print(f'RandomForestClassifier: class 조절 정확도 (accuracy) {accuracy}%')
    return model
