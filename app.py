import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the pre-trained model
model_path = 'fit_categorical_model.h5'  # Update the path if necessary
model = load_model(model_path)
print("Model loaded successfully.")

# Define the classes that the model will predict
classes = {0: 'fb', 1: 'idea', 2: 'yt'}

# Route for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Get the image file from the request
        file = request.files['image']
        filename = secure_filename(file.filename)

        # Create a temporary directory to save the uploaded image
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        # Preprocess the image for prediction
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_class_index]
        probability = float(prediction[0][predicted_class_index])

        # Clean up the temporary file
        os.remove(filepath)

        # Return the prediction results
        return jsonify({'class': predicted_class, 'probability': probability})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
