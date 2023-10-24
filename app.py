from flask import Flask, render_template, request, jsonify
import io
import os
import numpy as np
from numpy import asarray
from PIL import Image
import base64
import glob
import requests

from shuo_clip2 import get_response

app = Flask(__name__)


## Added
# @app.get("/image")
def render_frame(arr: np.ndarray):
    mem_bytes = io.BytesIO()
    img = Image.fromarray(arr)
    img.save(mem_bytes, 'JPEG')
    mem_bytes.seek(0)
    img_base64 = base64.b64encode(mem_bytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)
    return uri
## end added


@app.get("/image")
def image():
    image_names = request.args.get("image_names")
    image_names_list = image_names.split(", ")
    img = []
    for file in glob.glob('./data/image/*.*'):
        title = os.path.basename(file)
        if any(image_name in title for image_name in image_names_list):
            image = asarray(Image.open(file))
            img.append({'src': render_frame(image), 'title': title})

    return render_template("main.html", images=img)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    print(response)

    # Get images
    image_response = requests.get("http://127.0.0.1:5000/image", params={"image_names": response})
    if image_response.status_code == 200:
        print("images uploaded successfully")

    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
