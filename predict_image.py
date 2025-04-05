# Import tools
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained CNN
model = load_model(r"C:\Users\NIHA\OneDrive\Desktop\braintumorproject\trained_brain_tumor_cnn.h5")

# Load and prepare the image
img_path = r"C:\Users\NIHA\OneDrive\Desktop\braintumorproject\test_image.jpg"  # Update if you named it differently
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0  # Normalize to 0-1
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)
result = "Tumor" if prediction[0] > 0.5 else "No Tumor"
confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]
print(f"Prediction: {result} (Confidence: {confidence:.2f})")