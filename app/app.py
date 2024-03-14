from flask import Flask, request

from scripts.idefics import run_inference
from pathlib import Path

UPLOAD_FOLDER = './static/images/'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/ebm-url", methods=['POST'])
def process_image():

    info = request.get_json()
    img_url = info["img"]

    # print(img)
    food_groups, pred = run_inference(img_url, True)

    output = {"food_groups": food_groups, "prediction": pred}

    return f"Img URL is: {str(output)}", 200


@app.route('/ebm-image', methods=['POST'])
def upload():
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded file to a desired location
    file.save(Path(app.config['UPLOAD_FOLDER']) / str(file.filename))
    
    food_groups, pred = run_inference(file, False)

    output = {"food_groups": food_groups, "prediction": pred}

    return f"{str(output)}", 200
