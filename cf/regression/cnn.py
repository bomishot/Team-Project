import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def cnn_regression(df):
    y_target = df['Rings']
    X_features = df.drop('Rings', axis=1, inplace=False)

    # 데이터 정규화 (Normalization) -> 별 차이없고, 하면 정확도 조금늘고, 손실조금늘어남.
    #scaler = StandardScaler()
    #X_features_scaled = scaler.fit_transform(X_features)

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

    # CNN 모델 구성
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # 회귀 모델이므로 출력 뉴런은 1개
    ])

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')


    # 모델 학습
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # 테스트 데이터로 예측
    y_pred_train_cnn = model.predict(X_train).flatten()
    y_pred_test_cnn = model.predict(X_test).flatten()

    # MSE 계산
    mse_train_cnn = mean_squared_error(y_train, y_pred_train_cnn)
    acc_train_cnn = 1 - np.abs((y_pred_train_cnn - y_train) / y_train).mean()

    mse_test_cnn = mean_squared_error(y_test, y_pred_test_cnn)
    acc_test_cnn = 1 - np.abs((y_pred_test_cnn - y_test) / y_test).mean()

    print("CNN Regression:", '\n')
    print("************* train")
    print("MSE: ", mse_train_cnn)
    print("Accuracy: ", acc_train_cnn, '\n')

    print("************* test")
    print("MSE: ", mse_test_cnn)
    print("Accuracy: ", acc_test_cnn)

