import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

model_path = 'model.h5'

# Check if the model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' does not exist. Please place it in the correct directory.")

# Load the model from the h5 file
model_pipeline = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # Ensure consistent input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch size of 1
    img_array /= 255.0  # Normalize pixel values
    return img_array

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
    
    # Make prediction
    prediction = model_pipeline.predict(preprocessed_image)
    probability = prediction[0][0]
    
    if probability >= 0.5:
        result_text = f"You have Brain Tumor (Probability: {probability*100: .4f}%)"
    elif probability >= 0.009:
        result_text = f"No you do not have Brain Tumor(Probability: {probability*100:.4f}%)"
    else:
        result_text = "Not an MRI Scan"
    
    return render_template('result.html', filename=filename, result=result_text)

if __name__ == '__main__':
    app.run(debug=True)


