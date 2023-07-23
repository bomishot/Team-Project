import requests

url = 'http://localhost:5000/predict'
data = {
    'X_Minimum': 10,
    'Y_Minimum': 5,
    'Pixels_Areas': 2000,
    'Length_of_Conveyer': 15,
    'Steel_Plate_Thickness': 0.5,
    'Edges_Index': 0.7,
    'Empty_Index': 0.2,
    'Square_Index': 0.8,
    'Outside_X_Index': 0.6,
    'Edges_X_Index': 0.5,
    'Edges_Y_Index': 0.4,
    'Outside_Global_Index': 0.9,
    'LogOfAreas': 4.5,
    'Log_X_Index': 0.3,
    'Log_Y_Index': 0.2,
    'Orientation_Index': 0.1,
    'Luminosity_Index': -0.5,
    'SigmoidOfAreas': 0.6,
    'type': 2,
    'Total_Perimeter': 300,
    'Mean_of_Luminosity': 125
}

response = requests.post(url, json=data)
print(response.json())
