from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# 모델 로드하는 함수
def load_model():
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

@app.route('/', methods=['GET'])
def welcome():
    return '모델 예측 API에 오신 것을 환영합니다!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 모델 로드
            loaded_model = load_model()

            # JSON 데이터를 요청에서 가져옴
            data = request.json

            # JSON 데이터를 DataFrame으로 변환
            df = pd.DataFrame([data])

            # 모델을 사용하여 예측 수행
            predictions = loaded_model.predict(df)

            # 예측 결과를 리스트로 변환
            predictions_list = predictions.tolist()

            # JSON 형식으로 예측 결과 반환
            return jsonify({'predictions': predictions_list})
        except Exception as e:
            return jsonify({'error': '예측 중 오류가 발생했습니다.'}), 400
    else:
        return jsonify({'error': 'POST 메서드만 허용됩니다.'}), 405

# GET 메서드로 /predict에 접근 시 기본 화면 보여주기
@app.route('/predict', methods=['GET'])
def show_prediction_form():
    return '이곳은 모델 예측 API입니다. POST 메서드로 데이터를 보내주세요.'

if __name__ == '__main__':
    app.run(debug=True)
