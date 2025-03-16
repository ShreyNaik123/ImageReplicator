from flask import Flask, render_template, Response, request, jsonify, session,send_file
from model import Model, Model2, Model3
from utils import preprocess_image, view_image, image_to_tensor
import torch
import torch.nn as nn
import torch.optim as optim
import os
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from flask_cors import CORS
import json
import logging

matplotlib.use('Agg')



app = Flask(__name__)
app.secret_key = 'secretKey987'  # Replace with a real secret key
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def generate_images(image, num_epochs, learning_rate, model_choice):
    if(model_choice == "Model1"):
        
        model = Model()
    elif (model_choice == "Model2"):
        model = Model2()
    elif (model_choice == "Model3"):
        model = Model3()
        
    if torch.cuda.is_available():
        model = model.cuda()
    
    logger.info(f"Model initialized. CUDA available: {torch.cuda.is_available()}")

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        input_image = image.unsqueeze(0)
        if torch.cuda.is_available():
            input_image = input_image.cuda()
        

        output = model(input_image)
        loss = criterion(output, input_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

     
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        view_image(image.cpu(), title="Input")
        plt.subplot(1, 2, 2)
        view_image(output.squeeze(0).detach().cpu(), title=f"Epoch: {epoch + 1}")
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.getvalue()).decode()
        plt.close()


        yield {
            "epoch": epoch + 1,
            "loss": loss.item(),
            "image": f"data:image/png;base64,{img_base64}"
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    # Debug: Print form data

    # Retrieve values from the FormData
    learning_rate = float(request.form.get('learning_rate', 0))
    num_epochs = int(request.form.get('epochs', 0))
    model_choice = request.form.get('model', '')
    image_file = request.files.get('image')
    
   

    if not image_file:
        return jsonify({"error": "No image file uploaded"}), 400



    # Save the uploaded image to a temporary file
    image_path = "static/temp_image.jpg"
    image_file.save(image_path)

    # Verify the uploaded image
    try:
        with Image.open(image_path) as img:
            pass
    except Exception as e:
        return jsonify({"error": "Invalid image uploaded"}), 400

    # Store the path and other data in the session
    session['training_data'] = {
        'image_path': image_path,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'model_choice': model_choice
    }
    


    return jsonify({"message": "Training started"}), 202

@app.route('/stream', methods=['GET'])
def stream():
    training_data = session.get('training_data', None)
    if training_data is None:
      
        return jsonify({"error": "No training data available"}), 400

    image_path = training_data['image_path']
    learning_rate = training_data['learning_rate']
    num_epochs = training_data['num_epochs']
    model_choice = training_data['model_choice']

    # Load the image from the file
    try:

        
        image_tensor = preprocess_image(image_path).float() / 255.0
     
        
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": "Error processing image"}), 500

    def generate():
        for data in generate_images(image_tensor, num_epochs, learning_rate, model_choice):
            yield f"data: {json.dumps(data)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/save_model', methods=['GET'])
def save_model():
    # Retrieve the model choice from session or some other means
    model_choice = session.get('training_data', {}).get('model_choice', 'Model1')

    if model_choice == "Model1":
        model = Model()
    elif model_choice == "Model2":
        model = Model2()
    else:
        return jsonify({"error": "Invalid model choice"}), 400

    if torch.cuda.is_available():
        model = model.cuda()

    # Save the model weights
    model_weights_path = "static/model_weights.pth"
    torch.save(model.state_dict(), model_weights_path)

    # Provide the file for download
    return send_file(model_weights_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)