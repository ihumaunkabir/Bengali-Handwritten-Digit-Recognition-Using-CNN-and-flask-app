from flask import Flask, render_template, request

import re
import sys
import os
import base64

from scipy.misc import imresize, imread

from model.load import *

import numpy as np

sys.path.append(os.path.abspath('./model'))

# init flask app
app = Flask(__name__)

global model, graph
model, graph = init()


def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        #output.write(imgstr.decode('base64'))
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = imread('output.png', mode='RGB')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 3)
    with graph.as_default():
        out = model.predict(x)
        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == '__main__':
    app.run(debug=True)

