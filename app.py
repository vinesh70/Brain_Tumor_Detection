from flask import Flask, render_template, request, make_response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from fpdf import FPDF

# Load the pre-trained model
model = load_model('braintumor.h5')

def predict_tumor(image_path):
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        # Check if the image was successfully loaded
        if img is not None:
            # Resize the image
            img = cv2.resize(img, (150, 150))
            
            # Convert the image to a NumPy array
            img_array = np.array(img)
            
            # Reshape the array for model input
            img_array = img_array.reshape(1, 150, 150, 3)
            
            # Predict using the model
            prediction = model.predict(img_array)
            predicted_class_index = prediction.argmax()
            return predicted_class_index
        else:
            print("Error: Unable to read the image file:", image_path)
            return None
    except Exception as e:
        print("Error:", e)
        return None

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get image file from the request
        file = request.files['file']
        
        # Save the file temporarily
        file_path = 'temp_image.jpg'
        file.save(file_path)
        
        # Perform prediction
        predicted_class_index = predict_tumor(file_path)
        
        # Remove the temporary file
        os.remove(file_path)
        
        if predicted_class_index is not None:
            # Convert predicted class index to a regular Python integer
            predicted_class_index = int(predicted_class_index)
            
            # Define class labels and tumor types
            class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            
            # Get the predicted class label and tumor type
            predicted_class_label = class_labels[predicted_class_index]
            
            # Render the result page with predicted class label and tumor type
            return render_template('result.html', class_label=predicted_class_label)
        else:
            # Handle error case
            return render_template('result.html', error='Failed to predict')

# Route for downloading PDF
@app.route('/download_pdf')
def download_pdf():
    # Get the detected tumor label from the query parameters
    class_label = request.args.get('class_label')

    # Generate PDF content with the detected tumor label
    pdf_content = generate_pdf(class_label)

    # Create response with PDF content
    response = make_response(pdf_content)

    # Set content type and disposition for download
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=brain_tumor_detection_result.pdf'

    return response

def generate_pdf(class_label):
    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)

    # Add image and text to PDF
    pdf.image('temp_image.jpg', x = 10, y = 10, w = 180)
    pdf.cell(200, 10, txt = "Detected Tumor: " + class_label, ln = True, align = 'C')

    # Output PDF as bytes
    return pdf.output(dest='S').encode('latin1')


if __name__ == "__main__":
    app.run(debug=True, use_reloader = True)
