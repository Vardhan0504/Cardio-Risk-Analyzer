import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index1.html")

@flask_app.route("/login")
def Login():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    
    params = ['Age' , 'Sex' , 'Chest Pain Type' , 'Resting Blood Pressure' , 'Serum Cholestrol' , 'Fasting Blood Sugar' , 'Resting Electrocardiographic Results' , 'Maximum Heart Rate Achieved' , 'Exercise Induced Angina' , 'ST Depression Induced by Exercise Relative to Rest' , 'The Slope of The Peak Exercise ST Segment' , 'Number of Major Vessels Colored by Flourosopy' , 'Thallium Stress Test']
    float_features = []
    for param in params:
        float_features.append(float(request.form.get(param)))
        
    features = [np.array(float_features)]
    features = pd.DataFrame(features , columns = ['Age' , 'Sex' , 'Chest Pain Type' , 'Resting Blood Pressure' , 'Serum Cholestrol' , 'Fasting Blood Sugar' , 'Resting Electrocardiographic Results' , 'Maximum Heart Rate Achieved' , 'Exercise Induced Angina' , 'ST Depression Induced by Exercise Relative to Rest' , 'The Slope of The Peak Exercise ST Segment' , 'Number of Major Vessels Colored by Flourosopy' , 'Thallium Stress Test'])
    prediction = model.predict(features)
    if(prediction == 1):
        return render_template('emergency.html')
    else:
        return render_template('safe.html')


if __name__ == "__main__":
    flask_app.run(debug=True)