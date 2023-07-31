from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np

# Regression
def xgboost(df):
    y_target = df['Rings']
    X_features = df.drop('Rings',axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

    xgb_reg = XGBRegressor(n_estimators=400,
                        learning_rate=0.01,
                        colsample_bytree=0.8,
                        subsample=0.2,
                        gamma=0.2,
                        max_depth=5,
                        min_child_weight=5)

    xgb_reg.fit(X_train, y_train)

    y_pred = xgb_reg.predict(X_test)

    # 정확도 계산
    acc = np.mean(1 - np.abs((y_pred - y_test) / y_test))
    print('XGBOOST')
    print(f"Mean Accuracy: {acc:.4f}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2:.4f}")