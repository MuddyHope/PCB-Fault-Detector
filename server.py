
import os
from flask import Flask, render_template, request
from utils import get_model, predict


app = Flask(__name__)


get_model()

# Folder to save uploads (inside static)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def make_prediction():
    product = None
    file_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        # Save file safely
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        print("Saved:", file_path)

        # Call your prediction function
        product = predict(file_path)
        print("Prediction:", product)

    return render_template(
        'predict.html',
        product=product,
        user_image=file_path
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

