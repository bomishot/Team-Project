import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop

def rnn_regression(df):
    # 데이터를 'Rings' 오름차순으로 정렬

    y_target = df['Rings']
    X_features = df.drop(['Rings'], axis=1, inplace=False)

    # 데이터 정규화 (Normalization)
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features)

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X_features_scaled, y_target, test_size=0.2, random_state=156)

    # 데이터 형태를 RNN 입력 형태로 변환 (DataFrame to NumPy array)
    X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_rnn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # RNN 모델 구성
    model = keras.Sequential([
        layers.GRU(64, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
        layers.GRU(32, activation='relu', return_sequences=True),
        layers.GRU(16, activation='relu', return_sequences=True),
        layers.GRU(8, activation='relu'),
        layers.Dense(1, kernel_regularizer=l2(0.001))  # L2 규제를 적용한 출력층
    ])

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mean_squared_error')
       # 모델 학습
    model.fit(X_train_rnn, y_train, epochs=10, batch_size=16, verbose=1, validation_split=0.2)

    # 테스트 데이터로 예측
    y_pred_train_rnn = model.predict(X_train_rnn).flatten()
    y_pred_test_rnn = model.predict(X_test_rnn).flatten()

    # MSE 계산
    mse_train_rnn = mean_squared_error(y_train, y_pred_train_rnn)
    acc_train_rnn = 1 - np.abs((y_pred_train_rnn - y_train) / y_train).mean()

    mse_test_rnn = mean_squared_error(y_test, y_pred_test_rnn)
    acc_test_rnn = 1 - np.abs((y_pred_test_rnn - y_test) / y_test).mean()

    print("RNN Regression:", '\n')
    print("************* train")
    print("MSE: ", mse_train_rnn)
    print("Accuracy: ", acc_train_rnn, '\n')

    print("************* test")
    print("MSE: ", mse_test_rnn)
    print("Accuracy: ", acc_test_rnn)
