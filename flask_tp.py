from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf

app = Flask(__name__)
loaded_model_regression = tf.keras.models.load_model(r'./model.h5')
loaded_model_bynary = tf.keras.models.load_model(r'./model.h5')
loaded_model_multi = tf.keras.models.load_model(r'./model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about', methods=['POST'])
def about():
    return render_template('about.html')

@app.route('/error', methods=['POST'])
def error():
    return render_template('error.html')

@app.route('/regression', methods=['POST'])
def regression():
    return render_template('regression_data.html')

@app.route('/bynary', methods=['POST'])
def bynary():
    return render_template('binary_classification_data.html')

@app.route('/multi', methods=['POST'])
def multi():
    return render_template('multi_classification_data.html')

@app.route('/predict_reg', methods=['POST'])
def predict_reg():
    input_data = []
    for i in range(1, 8):
        input_field = f'input_data{i}'
        input_value = request.form.get(input_field)
        if input_value:
            input_data.append(int(input_value))
        else:
            return render_template('error.html')
    input_data = np.array([input_data])
    predicted_value = loaded_model_regression.predict(input_data)
    return render_template('result.html', predicted_value=predicted_value)

@app.route('/predict_bynary', methods=['POST'])
def predict_bynary():
    input_data = []
    for i in range(1, 8):
        input_field = f'input_data{i}'
        input_value = request.form.get(input_field)
        if input_value:
            input_data.append(int(input_value))
        else:
            return render_template('error.html')
    input_data = tf.constant([input_data])
    predicted_value = loaded_model_bynary.predict(input_data)
    return render_template('result.html', predicted_value=predicted_value)

@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    input_data = np.array([input_data])
    predicted_value = loaded_model_multi.predict(input_data)
    return render_template('result.html', predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
"""
X_dummy = tf.constant([[23.625, 29.94865, 5.688038, 35.98717, 146.5686, 82.39462, -0.2749, -1.12185]])
loaded_model = tf.keras.models.load_model('model.h5')
prediction = loaded_model.predict(X_dummy[:5])
print(prediction)
"""