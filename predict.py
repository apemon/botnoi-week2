from botnoi import cv

import pickle
import numpy as np
import pandas as pd

cal_df = pd.read_csv('cal-table.csv')
mymod = pickle.load(open('foodmodel-mobilenet.p','rb'))
def predict_image(img_url):
    a = cv.image(img_url)
    feat = a.getresnet50()
    probList = mymod.predict_proba([feat])[0]
    maxprobind = np.argmax(probList)
    prob = probList[maxprobind]
    outclass = mymod.classes_[maxprobind]
    result = {}
    result['class'] = outclass
    result['probability'] = prob
    food = cal_df.loc[cal_df['food_name'] == outclass]
    if len(food.index) > 0:
        result['cal'] = float(food['cal'].values[0])
        result['fat'] = float(food['fat'].values[0])
        result['protein'] = float(food['protein'].values[0])
        result['carbohydrate'] = float(food['carbohydrate'].values[0])
    return result