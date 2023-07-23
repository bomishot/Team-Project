from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def ridge(df):
    y_target = df['Rings']
    X_features = df.drop('Rings',axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

    ridge_reg = Ridge(alpha=0.00005)
    ridge_reg.fit(X_train, y_train)

    y_pred_train_ridge = ridge_reg.predict(X_train)
    y_pred_test_ridge = ridge_reg.predict(X_test)

    mse_train_ridge = mean_squared_error(y_train, y_pred_train_ridge)
    acc_train_ridge = 1 - abs((y_pred_train_ridge - y_train) / y_train).mean()

    mse_test_ridge = mean_squared_error(y_test, y_pred_test_ridge)
    acc_test_ridge = 1 - abs((y_pred_test_ridge - y_test) / y_test).mean()

    print("Ridge Regression:", '\n')
    print("************* train")
    print("MSE: ", mse_train_ridge)
    print("Accuracy: ", acc_train_ridge,'\n')

    print("************* test")
    print("MSE: ", mse_test_ridge)
    print("Accuracy: ", acc_test_ridge)