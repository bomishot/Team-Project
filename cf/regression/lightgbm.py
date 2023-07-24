from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import numpy as np

def lightgbm(df):
    y_target = df['Rings']
    X_features = df.drop('Rings',axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)
    X_train.columns = X_train.columns.str.replace(' ', '_') # lightbgm 돌리면 warning 많이 떠서해줌.
    X_test.columns = X_test.columns.str.replace(' ', '_')
    lgbm_reg = LGBMRegressor(colsample_bytree=0.8, learning_rate=0.01, n_estimators=500,
              n_jobs=-1, num_leaves=16, reg_lambda=10, subsample=0.8)
    lgbm_reg.fit(X_train, y_train)

    y_pred = lgbm_reg.predict(X_test)

        # 정확도 계산
    acc = np.mean(1 - np.abs((y_pred - y_test) / y_test))
    print('LightGBM : ')
    print(f"Mean Accuracy: {acc:.4f}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}")