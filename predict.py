from botnoi import cv
import pickle
import numpy as np

def predict_image(img_url):
    mymod = pickle.load(open('mymodel.p','rb'))
    a = cv.image(img_url)
    feat = a.getresnet50()
    probList = mymod.predict_proba([feat])[0]
    maxprobind = np.argmax(probList)
    prob = probList[maxprobind]
    outclass = mymod.classes_[maxprobind]
    result = {}
    result['class'] = outclass
    result['probability'] = prob
    return result