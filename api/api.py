
from flask import Flask, jsonify, request
from inference_one import fetch_features_one

app = Flask(__name__)

@app.route('/prediction',methods=['POST'])
def make_prediction():

    return 