import os
import warnings
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template
from keras.models import load_model # type: ignore
import base64
from io import BytesIO

# Silence logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Flask setup
app = Flask(__name__)

# Load model and class names
model = load_model('dog_breed_model.h5')
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

def preprocess_image(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    opencv_image = cv2.resize(opencv_image, (224, 224))
    opencv_image = opencv_image.astype('float32') / 255.0
    opencv_image = np.expand_dims(opencv_image, axis=0)
    return opencv_image

def convert_image_to_base64(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_data = None
    if request.method == 'POST':
        if 'dog_image' in request.files:
            file = request.files['dog_image']
            if file:
                img_bytes = file.read()
                processed_img = preprocess_image(img_bytes)
                preds = model.predict(processed_img)
                predicted_label = CLASS_NAMES[np.argmax(preds)]
                prediction = f"The Dog Breed is {predicted_label}"
                image_data = convert_image_to_base64(img_bytes)
    return render_template('index.html', prediction=prediction, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
