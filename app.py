from flask import Flask, jsonify, request

import predict

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'ok'

@app.route('/predict')
def predict_food():
    url = request.args.get('img')
    predict_result = predict.predict_image(url)
    return predict_result

if __name__ == '__main__':
    app.run()