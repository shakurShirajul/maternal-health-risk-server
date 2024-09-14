import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

knnModel = pickle.load(open('knneighbours.pkl', 'rb'))


@app.route('/')
def home():
   return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    values_list = [list(request.json.values())]
    final_input = (np.array(values_list).reshape(1, -1))
    
    print(final_input)
    
    output = knnModel.predict(final_input)[0]
    print(output)
    return jsonify({'result':output})


if __name__ == '__main__':
    app.run(debug=True)
