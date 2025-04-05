# Import tools
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained CNN
model = load_model(r"C:\Users\NIHA\OneDrive\Desktop\braintumorproject\trained_brain_tumor_cnn.h5")

# Set up image feeder for test data
datagen = ImageDataGenerator(rescale=1./255)

# Load test images
test_generator = datagen.flow_from_directory(
    r"C:\Users\NIHA\OneDrive\Desktop\braintumorproject\organized_data\test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Keep order for predictions
)

# Make predictions
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()  # 0 or 1

# Get true labels
true_classes = test_generator.classes

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Print confusion matrix as text
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(cm)
print("[[True No Tumor, False Tumor],")
print(" [False No Tumor, True Tumor]]")

# Print detailed report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=['No Tumor', 'Tumor']))