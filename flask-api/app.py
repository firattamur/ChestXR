import os

from flask import Flask, render_template, request, redirect

# from predict import predict_image
# from utils

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():

    if request.method == "POST":

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get("file")

        if not file:
            return

        image_bytes = file.read()
        class_name, confidence = predict_image(image_bytes=image_bytes)

        return render_template("result.html", class_name=class_name, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))

