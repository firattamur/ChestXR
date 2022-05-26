import os

from utils.utils_commandline import load_config
from flaskapi.predict import predict_image
from flask import Flask, render_template, request, redirect

# simple flask app
app = Flask(__name__, static_folder="./static")

# load configurations
config = load_config()


@app.route("/", methods=["GET", "POST"])
def home():

    

    if request.method == "POST":

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get("file")

        print(file)

        if not file:
            return

        image_bytes = file.read()
        class_name, confidence = predict_image(config, image_bytes=image_bytes)

    return render_template("index.html", class_name=10, confidence=10)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))

