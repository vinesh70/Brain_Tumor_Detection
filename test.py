import cv2
import numpy as np
from tensorflow.keras.models import load_model

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

# Load the pre-trained model
model = load_model('braintumor.h5')

# Path to the directory containing image datasets
image_directory = 'datasets/Training/'

# Test for different tumor types
tumor_types = ['pituitary_tumor', 'no_tumor', 'glioma_tumor', 'meningioma_tumor']

for tumor_type in tumor_types:
    image_path = f"{image_directory}/{tumor_type}/"
    if tumor_type == 'no_tumor':
        image_path += 'image(317).jpg'  # For 'no_tumor' class
    elif tumor_type == 'meningioma_tumor':
        image_path += 'm2 (89).jpg'  # For 'meningioma_tumor' class
    elif tumor_type == 'glioma_tumor':
        image_path += 'gg (11).jpg'  # For 'glioma_tumor' class
    else:
        image_path += f'{tumor_type[0]} (1).jpg'  # For other tumor classes
    
    predicted_class_index = predict_tumor(image_path)
    if predicted_class_index is not None:
        print(f"Predicted class index for {tumor_type}: {predicted_class_index}")


if predicted_class_index == 0:
    print("You have a Glioma Tumor")
elif predicted_class_index == 1:
    print("You have a Meningioma Tumor")
elif predicted_class_index == 2:
    print("You have No Tumor")
elif predicted_class_index == 3:
    print("You have a Pituitary Tumor")