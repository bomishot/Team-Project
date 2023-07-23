from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def lasso(df):
    y_target = df['Rings']
    X_features = df.drop('Rings',axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

    lasso_reg = Lasso(alpha=0.001)
    lasso_reg.fit(X_train, y_train)
    
    y_pred_train_lasso = lasso_reg.predict(X_train)
    y_pred_test_lasso = lasso_reg.predict(X_test)

    mse_train_lasso = mean_squared_error(y_train, y_pred_train_lasso)
    acc_train_lasso = 1 - abs((y_pred_train_lasso - y_train) / y_train).mean()

    mse_test_lasso = mean_squared_error(y_test, y_pred_test_lasso)
    acc_test_lasso = 1 - abs((y_pred_test_lasso - y_test) / y_test).mean()

    print("Lasso Regression:", '\n')
    print("************* train")
    print("MSE: ", mse_train_lasso)
    print("Accuracy: ", acc_train_lasso, '\n')

    print("************* test")
    print("MSE: ", mse_test_lasso)
    print("Accuracy: ", acc_test_lasso)
