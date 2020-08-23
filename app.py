from flask import Flask, jsonify, request

import predict

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'ok'

@app.route('/predict')
def predict_food():
    url = request.args.get('url')
    predict_result = predict.predict_image(url)
    result = {}
    result['predict'] = predict_result
    result['food'] = predict_result['class']
    result['cal'] = 808
    return result

if __name__ == '__main__':
    app.run()