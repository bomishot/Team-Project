from model_utils import load_dataset, RandomForestClassifier

def main():
    # 데이터셋 불러오기
    dataset = load_dataset()
    RandomForestClassifier(dataset)

if __name__ == "__main__":
    main()

