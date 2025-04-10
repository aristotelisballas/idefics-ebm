from flask import Flask, request

from scripts.idefics import run_inference
# from scripts.food_llama import run_inference
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
    food_groups, pred = run_inference(img_url, True,True)

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
    # file.save(Path(app.config['UPLOAD_FOLDER']) / str(file.filename))
    
    food_groups, pred = run_inference(file, False,True)

    output = {"food_groups": food_groups, "prediction": pred}

    return f"{str(output)}", 200


@app.route('/ebm-image-updated', methods=['POST'])
def upload_udpated():
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded file to a desired location
    file_path = Path(app.config['UPLOAD_FOLDER']) / str(file.filename)
    file.save(file_path)
    
    # Reopen the saved file so that Image.open can properly read it
    with open(file_path, 'rb') as f:
        food_groups, pred = run_inference(f, False, False)

    output = {"food_groups": food_groups, "prediction": pred}

    return f"{str(output)}", 200

