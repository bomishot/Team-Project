from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

def rnn_classification(df):
    # X, y
    target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 훈련 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # time steps 설정
    time_steps = 1  # 예시로 1로 설정했지만, 시퀀스 데이터의 길이에 맞게 조정 가능

    # shape 재구성 (samples, time steps, features)
    X_train = X_train.reshape(X_train.shape[0], time_steps, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], time_steps, X_test.shape[1])

    # Modeling
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 훈련
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')