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
    img = request.files['img_File'].stream

    pil_image = Image.open(img).convert('RGB')

    # pil_image.show()

    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    name = classify(open_cv_image)
    # cv2.imshow("faces", open_cv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return "<h3>"+name+"</h3>"


app.run()
