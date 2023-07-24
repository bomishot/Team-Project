from model_utils import load_dataset_regression, load_dataset_binary_classification, load_dataset_multi_classification_woo, load_dataset_multi_classification_gang
from model_utils import rnn_regression, binary_model, train_random_forest_classifier, rnn_classification
import time

def main():
    # 1) 데이터셋 불러오기
    #reg_dataset = load_dataset_regression()
    binary_dataset = load_dataset_binary_classification()
    #multi_dataset_woo = load_dataset_multi_classification_woo()
    #multi_dataset_gang = load_dataset_multi_classification_gang()

    # 2) 각 데이터별 성능 좋은 모델 불러오기
    start_time = time.time()
    print('~~~~~~~~~~~~~~~~~~~~','\n','Regression')
    #rnn_regression(reg_dataset)
    # [76s] (epoch20) MSE:4.86, accuracy:83% 

    print('~~~~~~~~~~~~~~~~~~~~','\n','Binary Classification')
    binary_model(binary_dataset) 
    # 69번 epoch 너무 오래 걸려요!

    print('~~~~~~~~~~~~~~~~~~~~','\n','Multi Classification - woo')
    #train_random_forest_classifier(multi_dataset_woo)
    # [4s] accuracy good 93%

    print('~~~~~~~~~~~~~~~~~~~~','\n','Multi Classification - gang')
    #rnn_classification(multi_dataset_gang)
    # [1m 9s] loss:  0.66 , accuracy:76%

    end_time = time.time()
    print('Total Time : ', end_time-start_time, 'seconds')


if __name__ == "__main__":
    main()

