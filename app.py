import pickle
import os
from flask import Flask, request, app, jsonify, url_for,render_template

import pandas as pd
import numpy as np

app = Flask(__name__)

#loading the model and scaler

regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    # print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    # new_data = scaler.transform(np.array(list(data.values())).reshape(-1,1))
    new_data = scaler.transform([np.array(list(data.values()))])
    output = regmodel.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data  = [float(x) for x in request.form.values()]
    # print(data)
    final_input = scaler.transform([np.array(list(data))])
    # print(final_input)
    output = regmodel.predict(final_input)
    return render_template('home.html', prediction_text = f"The House price is {output[0]}")

if __name__ =="__main__":
    app.run(debug=True)











