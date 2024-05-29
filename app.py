import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

model_path = 'model.pth'  # Path to your PyTorch model file

# Load the model from the PyTorch file
model = torch.load(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    # Load and preprocess the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to match the input size of the model
    img = img / 255.0  # Normalize pixel values
    img = np.transpose(img, (2, 0, 1))  # Transpose dimensions to match PyTorch format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = torch.tensor(img, dtype=torch.float32)  # Convert to PyTorch tensor
    return img

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create the uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(file_path)
            return redirect(url_for('display', filename=filename))
    return render_template('upload.html')

@app.route('/display/<filename>')
def display(filename):
    return render_template('display.html', filename=filename)

@app.route('/result/<filename>')
def result(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    preprocessed_image = preprocess_image(img_path)
    
    # Make prediction using the loaded model
    prediction = model(preprocessed_image)
    probability = torch.sigmoid(prediction).item()
    
    if probability >= 0.5:
        result_text = f"You have Brain Tumor (Probability: {probability*100: .4f}%)"
    elif probability >= 0.001:
        result_text = f"No you do not have Brain Tumor(Probability: {probability*100:.4f}%)"
    else:
        result_text = "Not an MRI Scan"
    
    return render_template('result.html', filename=filename, result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
