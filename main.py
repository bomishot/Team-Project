from model_utils import load_dataset_regression, load_dataset_binary_classification, load_dataset_multi_classification
from model_utils import binary_model, xgboost, multi_classification
import time

def main():
    # 1) 데이터셋 불러오기
    reg_dataset = load_dataset_regression()
    binary_dataset = load_dataset_binary_classification()
    multi_dataset = load_dataset_multi_classification()

    # 2) 각 데이터별 성능 좋은 모델 불러오기
    start_time = time.time() # 각 모델 별 실행 시간을 확인하세요.
    print('~~~~~~~~~~~~~~~~~~~~','\n','Regression')
    xgboost(reg_dataset)
    # [0.8s]Mean Accuracy: 0.8608, MSE: 3.9293, R2 Score: 0.5441

    print('~~~~~~~~~~~~~~~~~~~~','\n','Binary Classification')
    binary_model(binary_dataset) 
    # [2s] BinaryClassifier: class 조절 정확도 (accuracy) 97.7%

    print('~~~~~~~~~~~~~~~~~~~~','\n','Multi Classification ')
    multi_classification(multi_dataset)
    # [16s] RandomForestClassifier: class 조절 정확도 (accuracy) 95.23%

    end_time = time.time()
    print('Total Time : ', end_time-start_time, 'seconds')


if __name__ == "__main__":
    main()

