from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('xgb_random_search.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    unit_weight = float(request.form['unit_weight'])
    cohesion = float(request.form['cohesion'])
    friction = float(request.form['friction'])
    slope_angle = float(request.form['slope_angle'])
    slope_height = float(request.form['slope_height'])
    pore_pressure = float(request.form['pore_pressure'])

    new_data = np.array([[unit_weight, cohesion, friction,
                          slope_angle, slope_height, pore_pressure]])

    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)

    result = "STABLE" if prediction[0] == 1 else "FAILURE"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)