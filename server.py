import flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import cv2
from PIL import Image
import numpy as np
from main_file import classify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

CORS(app)


@app.route('/', methods=['POST'])
def home():
    # Reading image stream
    img = request.files['img_File'].stream
    pil_image = Image.open(img).convert('RGB')
    # converting the pil_image into an array
    open_cv_image = np.array(pil_image)
    # converting the RGB TO BGR for opencv
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    name = classify(open_cv_image)
    return "<h3>"+name+"</h3>"


app.run()
