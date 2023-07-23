from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib

df = pd.read_csv("preprocessed_data.csv")
# X는 독립 변수 데이터, y는 종속 변수 데이터(타겟 데이터)
X = df.drop("targets", axis=1)
y = df['targets']
# 데이터를 훈련용(train)과 테스트용(test)으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
model_pred = model.predict(X_test)
print(classification_report(y_test, model_pred))
print(f'RandomForestClassifier: class 조절 정확도 (accuracy) {round(model.score(X_test, y_test)*100, 1)}%')

# 모델을 model.pkl로 저장
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
