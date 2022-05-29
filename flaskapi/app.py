import base64
import os
import time

from utils.utils_commandline import load_config
from flaskapi.predict import predict_image
import io
from PIL import Image
from flask_cors import CORS
from flask import Flask, render_template, request, redirect, url_for


# simple flask app
app = Flask(__name__, static_folder="./static")
CORS(app)

# load configurations
config = load_config()


@app.route("/", methods=["GET", "POST"])
def home():

    image = Image.open("../flaskapi/static/demo/demo.jpeg").convert('RGB')

    if request.method == "POST":

        if 'file' not in request.files:
            return redirect(url_for("/"))

        file = request.files.get("file")

        if not file:
            return

        image_bytes = file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image.save("../flaskapi/static/demo/demo.jpeg")

    data = io.BytesIO()
    image.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    response = predict_image(config, image=image)

    if len(response.keys()) != 0:

        top1_data = io.BytesIO()
        image = response["top1_image"].convert('RGB')
        image.save(top1_data, "JPEG")
        top1_image_encoded = base64.b64encode(top1_data.getvalue())

        top2_data = io.BytesIO()
        image = response["top2_image"].convert('RGB')
        image.save(top2_data, "JPEG")
        top2_image_encoded = base64.b64encode(top2_data.getvalue())

        top3_data = io.BytesIO()
        image = response["top3_image"].convert('RGB')
        image.save(top3_data, "JPEG")
        top3_image_encoded = base64.b64encode(top3_data.getvalue())

        return render_template(

                        "index.html",
                        input_image=encoded_img_data.decode('utf-8'),

                        top1_image=top1_image_encoded.decode('utf-8'),
                        top1_category=response["top1_category"],

                        top2_image=top2_image_encoded.decode('utf-8'),
                        top2_category=response["top2_category"],

                        top3_image=top3_image_encoded.decode('utf-8'),
                        top3_category=response["top3_category"]

                        )
    else:
        return render_template("index.html",
                               input_image=encoded_img_data.decode('utf-8'))


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))

