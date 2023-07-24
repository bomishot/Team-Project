import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def binary_model(df):
    # X, y 분리
    target = 'target_class'
    X = df.drop(columns = 'target_class')
    y = df[target]

    # Scaling
    scaler = MinMaxScaler()
    X[:] = scaler.fit_transform(X[:])
    
    # Train, Validation, Test Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2)
    
    # Modeling
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape = (8,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                metrics=['accuracy'])

    # Train
    results = model.fit(X_train, y_train,
                        epochs=69,  # gridsearch
                        batch_size = 1, 
                        validation_data=(X_val, y_val))   

    # Test
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)

    # Classification Report
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


    
