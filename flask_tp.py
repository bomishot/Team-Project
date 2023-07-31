from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import StandardScaler

"""
Flask를 사용하여
세 가지 다른 모델로 데이터를 예측 및 그 결과를 웹 페이지로 표시 
localhost:5000에서 동작하도록 구현함.
"""

app = Flask(__name__)

# loading model
loaded_model_regression = pickle.load(open('./models/regression.pkl', "rb"))
loaded_model_bynary = pickle.load(open('./models/binary.pkl', "rb"))
loaded_model_multi = pickle.load(open('./models/multi.pkl', "rb"))

# 홈페이지 route
@app.route('/')
def home():
    return render_template('index.html')

# 프로젝트 소개 페이지 route
@app.route('/about', methods=['GET', 'POST'])
def about():
    title = "프로젝트 소개"
    content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed eu dapibus urna, et rhoncus ipsum. Nulla facilisi. Vestibulum scelerisque ac odio non fermentum. Vivamus nec massa id tellus semper elementum. Duis ullamcorper vestibulum sem, eu dapibus elit consequat eu. Sed nec varius ex, vitae vehicula est. Vestibulum nec metus elit. Vestibulum vel justo vel quam volutpat euismod. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Suspendisse a lectus lorem. Etiam ullamcorper ligula eget risus dictum varius. Nunc non hendrerit elit."
    return render_template('about.html', title = title, content = content)

# 오류 페이지 route
@app.route('/error', methods=['POST'])
def error():
    return render_template('error.html')

# Regression Model route
@app.route('/regression', methods=['POST'])
def regression():
    return render_template('regression_data.html')

# Binary-Classification Model route
@app.route('/bynary', methods=['POST'])
def bynary():
    return render_template('binary_classification_data.html')

# Multi-Classification Model route
@app.route('/multi', methods=['POST'])
def multi():
    return render_template('multi_classification_data.html')

# Regression model predict result route
@app.route('/predict_reg', methods=['POST'])
def predict_reg():
    # input data preprocessing
    sex_value = request.form['column1']
    if sex_value == 'M':
        sex_value = 0
    elif sex_value == 'F':
        sex_value = 1
    elif sex_value == 'I':
        sex_value = 2
    input_data = []
    input_data.append(sex_value)
    for i in range(2, 9):
        input_field = f'input_data{i}'
        input_value = request.form.get(input_field)
        if len(input_value) > 0 and not bool(re.search(r'[^0-9.-]', input_value)):
            input_data.append(float(input_value))
    
    # predict with preprocessed data, print result
    if len(input_data) != 8:
        return render_template('error.html')
    else:
        input_data = np.array([input_data])
        if input_data[0][4] < input_data[0][5] + input_data[0][6] + input_data[0][7]:
            predicted_value = r"조개, 내장(Viscera), 껍질(Shell) 무게의 합이 전체 무게(Whole)를 초과했습니다."
        else:
            predicted_value = loaded_model_regression.predict(input_data)
            predicted_value = round(predicted_value[0])
        return render_template('regression_data.html', predicted_value=predicted_value, input_data=input_data)

# Binary Classification predict result route
@app.route('/predict_bynary', methods=['POST'])
def predict_bynary():
    # input data preprocessing
    input_data = []
    for i in range(1, 9):
        input_field = f'input_data{i}'
        input_value = request.form.get(input_field)
        if len(input_value) > 0 and not bool(re.search(r'[^0-9.-]', input_value)):
            input_data.append(float(input_value))

    # predict with preprocessed data, print result
    if len(input_data) != 8:
        return render_template('error.html')
    else:
        input_data_test = np.array([input_data])
        input_data = np.array([input_data])
        # 저장된 스케일러 객체 로드
        with open('binary_class_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            input_data = scaler.transform(input_data)

        a = loaded_model_bynary.predict(input_data)
        prediction_proba = loaded_model_bynary.predict_proba(input_data)
        for idx, (pred, proba) in enumerate(zip(loaded_model_bynary.predict(input_data), prediction_proba)):
            if pred == 1:
                print(f"Data point {idx+1}: Predicted Class: 1 (Pulsar), Probability: {proba[1]*100:.2f}%")
                c = round(proba[1]*100, 2)
            else:
                print(f"Data point {idx+1}: Predicted Class: 0 (NOt Pulsar), Probability: {proba[0]*100:.2f}%")
                b = round(proba[0]*100, 2)
        if a == 0:
            predicted_value = "비펄서"
            percentage = b
        elif a == 1:
            predicted_value = "펄서"
            percentage = c
        return render_template('binary_classification_data.html', predicted_value=predicted_value, percentage = percentage, input_data=input_data_test)

# Multi-Classification predict result route
@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    input_data = []
    for i in range(1, 28):
        input_field = f'input_data{i}'
        input_value = request.form.get(input_field)
        if len(input_value) > 0 and not bool(re.search(r'[^0-9.-]', input_value)):
            input_data.append(float(input_value))

    # predict with preprocessed data, print result
    if len(input_data) != 27:
        return render_template('error.html')
    else:
        # 입력 데이터 중 일부 삭제
        del input_data[26]
        del input_data[25]
        del input_data[23]
        del input_data[20]
        del input_data[19]
        del input_data[18]
        del input_data[16]
        del input_data[15]
        del input_data[12]
        del input_data[11]
        del input_data[9]
        del input_data[8]
        del input_data[7]
        del input_data[6]
        del input_data[5]
        del input_data[3]
        del input_data[2]
        del input_data[1]
        input_data = np.array([input_data])
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)
        
        a = loaded_model_multi.predict(input_data)
        result = "예측 결과"
        # 갑판 유형 출력 및, 유형에 따른 해결과정 
        if a == 0:
            predicted_value = 'Pastry'
            story = "강판의 표면에 빵 모양의 파손이 발생한 결함을 의미함. 강판 표면이 부서져서 빵처럼 보이는 경우"
            explain = "- 품질 관리 및 검사: 제조 과정에서 스테인레스 강판의 표면에 페이스트리 결함이 발생하는 원인을 파악하고, 품질 관리팀이 이를 지속적으로 모니터링하여 결함이 최소화되도록 합니다."
            explain2 = "- 생산 공정 개선: 페이스트리 결함이 발생하는 생산 과정을 분석하여 개선점을 찾습니다. 생산 과정을 향상시키고 페이스트리 결함의 발생을 줄이는 데에 집중합니다."
            explain3 = "- 코팅 또는 보호 처리: 페이스트리 결함이 발생하지 않도록 강판의 표면을 적절히 보호합니다. 코팅 또는 보호 필름을 사용하여 표면의 내구성을 향상시키고 결함 발생을 예방합니다."
            explain4 = "- 품질 향상을 위한 지속적인 노력: 페이스트리 결함과 같은 결함들을 감지하고 이해하는데에 지속적인 노력을 기울입니다. 품질 향상을 위한 프로세스 개선과 품질 관리를 수행합니다."
        elif a == 1:
            predicted_value = 'Z_Scratch'
            story = """강판 표면에 Z자 모양으로 긁힘 또는 스크래치가 생긴 결함을 의미합니다. Z자 형태의 긁힌 흔적이 있는 경우를 말합니다."""
            explain = "- 표면 검사: Z자 모양 긁힘을 감지하고 표면에 이상이 있는지 확인하는 정기적인 검사를 시행합니다."
            explain2 = "- 표면 보호: 강판의 표면을 보호하기 위해 적절한 방법으로 코팅 또는 보호 필름을 적용합니다."
            explain3 = ""
            explain4 = ""
        elif a == 2:
            predicted_value = 'K_Scatch'
            story = """강판 표면에 K자 모양으로 긁힘 또는 스크래치가 생긴 결함을 의미합니다. K자 형태의 긁힌 흔적이 있는 경우를 말합니다."""
            explain = "- 표면 검사: K자 모양 긁힘을 감지하고 표면에 이상이 있는지 확인하는 정기적인 검사를 시행합니다."
            explain2 = "- 표면 보호: 강판의 표면을 보호하기 위해 적절한 방법으로 코팅 또는 보호 필름을 적용합니다."
            explain3 = ""
            explain4 = ""
        elif a == 3:
            predicted_value = 'Stains'
            story = """강판 표면에 얼룩 모양의 더러움이 생긴 결함을 의미합니다. 표면에 얼룩이 묻어있거나 얼룩 모양의 더러움이 있는 경우를 말합니다."""
            explain = "- 청소 및 유지 관리: 표면에 얼룩이 발생하는 원인을 파악하고 적절한 청소 및 유지 관리를 시행합니다."
            explain2 = "- 표면 보호: 강판의 표면을 보호하기 위해 적절한 코팅 또는 보호 필름을 적용하여 얼룩 발생을 최소화합니다."
            explain3 = ""
            explain4 = ""
        elif a == 4:
            predicted_value = 'Dirtiness'
            story = "강판 표면에 더러운 자국이 있는 결함을 의미합니다. 표면에 먼지나 오염물이 묻어있는 경우를 말합니다."
            explain = "- 청소 및 유지 관리: 표면에 더러운 자국이 발생하는 원인을 파악하고 적절한 청소 및 유지 관리를 시행합니다."
            explain2 = "- 표면 보호: 강판의 표면을 보호하기 위해 적절한 코팅 또는 보호 필름을 적용하여 더러움 발생을 최소화합니다."
            explain3 = ""
            explain4 = ""
        elif a == 5:
            predicted_value = 'Bumps'
            story = """강판 표면에 덤불 모양으로 돌출된 결함을 의미합니다. 표면에 덤불 모양으로 돌출된 부분이 있는 경우를 말합니다."""
            explain = "- 제조 공정 개선: 강판 생산 공정에서 덤불 모양의 결함이 발생하는 원인을 분석하고, 공정을 개선하여 덤불 모양의 발생을 줄입니다."
            explain2 = "- 표면 보호: 덤불 모양이 발생하지 않도록 강판 표면을 보호하는 적절한 방법을 채택합니다."
            explain3 = ""
            explain4 = ""
        else:
            predicted_value = 'Other_Faults'
            story = """위에 언급된 6가지 유형 외에 다른 종류의 결함을 의미합니다. 즉, 다양한 유형의 결함들을 통칭하는 범주입니다."""
            explain = "- 특정 결함들에 대한 전문 지식과 품질 관리: 다양한 유형의 결함들에 대해 전문 지식을 가진 품질 관리팀과 협력하여 적절한 대응 방안을 마련합니다."
            explain2 = "- 품질 향상을 위한 지속적인 노력: 기타 결함들을 감지하고 이해하는데에 지속적인 노력을 기울이며, 품질 향상을 위한 프로세스 개선과 품질 관리를 수행합니다."
            explain3 = ""
            explain4 = ""
        return render_template('result.html', predicted_value=predicted_value, result = result, story = story, explain = explain, explain2 = explain2, explain3 = explain3, explain4 = explain4)

if __name__ == '__main__':
    app.run(debug=True)