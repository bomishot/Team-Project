from model_utils import load_dataset, linear_regression, ridge, lasso, xgboost, lightgbm, cnn_regression, rnn_regression

def main():
    # 데이터셋 불러오기
    dataset = load_dataset()

    # Modeling
    # ML Model
    # linear_regression(dataset)
    # ridge(dataset)
    # lasso(dataset)
    # 이 중, lasso 성능이 가장 good
    # (test) MSE : 4.814, Accuracy : 0.839
    #xgboost(dataset)
    # lightgbm(dataset)
    # 5개중, xgboost가 성능 가장 좋음.
    # (test) MSE : 4.644, Accuracy : 0.855

    # DL Model
    #cnn_regression(dataset)
    rnn_regression(dataset)
    # 현재 rnn 성능  : (test) MSE : 4.50, Accurach : 0.862
    #ensemble(dataset)

if __name__ == "__main__":
    main()

