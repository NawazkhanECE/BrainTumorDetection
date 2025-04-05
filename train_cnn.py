# Import tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf

# Build the CNN fresh
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile it fresh
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up image feeder
datagen = ImageDataGenerator(rescale=1./255)

# Load training images
train_generator = datagen.flow_from_directory(
    r"C:\Users\NIHA\OneDrive\Desktop\braintumorproject\organized_data\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load validation images
val_generator = datagen.flow_from_directory(
    r"C:\Users\NIHA\OneDrive\Desktop\braintumorproject\organized_data\val",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Train the CNN
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Plot results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save the trained CNN
model.save(r"C:\Users\NIHA\OneDrive\Desktop\braintumorproject\trained_brain_tumor_cnn.h5")
print("Training done and model saved!")