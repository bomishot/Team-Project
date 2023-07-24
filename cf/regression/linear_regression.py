from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def linear_regression(df):
    y_target = df['Rings']
    X_features = df.drop('Rings',axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

    lr_reg = LinearRegression()
    lr_reg.fit(X_train, y_train)

    y_pred_train_lr = lr_reg.predict(X_train)
    y_pred_test_lr = lr_reg.predict(X_test)

    mse_train_lr = mean_squared_error(y_train, y_pred_train_lr)
    acc_train_lr = 1 - abs((y_pred_train_lr - y_train) / y_train).mean()

    mse_test_lr = mean_squared_error(y_test, y_pred_test_lr)
    acc_test_lr = 1 - abs((y_pred_test_lr - y_test) / y_test).mean()

    print("Linear Regression:", '\n')
    print("************* train")
    print("MSE: ", mse_train_lr)
    print("Accuracy: ", acc_train_lr, '\n')

    print("************* test")
    print("MSE: ", mse_test_lr)
    print("Accuracy: ", acc_test_lr)
