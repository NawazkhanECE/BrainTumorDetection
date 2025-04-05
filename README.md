# Brain Tumor Detection Using CNN

## Project Overview
This is a machine learning project I built to detect brain tumors in MRI images using a Convolutional Neural Network (CNN). I completed this as a 3rd-year ECE student to prepare for placements. The project achieved 88.46% accuracy on test data!

## What I Did
- Used Python and TensorFlow to create a CNN.
- Trained it on 245 brain MRI images (154 with tumors, 91 without).
- Tested it on new images, getting 88.46% accuracy and 100% tumor detection.
- Applied it to a real-world MRI image and got the correct result.

## Files
- `train_cnn.py`: Code to build and train the CNN.
- `test_cnn.py`: Code to test the CNN on 26 images.
- `predict_image.py`: Code to predict on a random image.
- `trained_brain_tumor_cnn.h5`: The saved trained model (not uploaded due to size, available locally).
- `test_image.jpg`: The image used for real-world testing.

## Dataset
I used a dataset of 245 MRI images (not uploaded due to size and copyright). It was split into training (195), validation (24), and test (26) sets.

## Skills Learned
- Python programming (loops, functions).
- Machine learning basics (CNNs, training, testing).
- Debugging and problem-solving.

## Future Improvements
- Add more images to reduce overfitting.
- Use dropout or data augmentation for better accuracy.

## How to Run
1. Install Python and libraries (TensorFlow, NumPy).
2. Download a brain MRI image and update `predict_image.py` with the path.
3. Run `python predict_image.py` to see the prediction.

## Contact
Created by Nawaz khan. GitHub: [NawazkhanECE]. Email: khannawazkn2004@gmail.com.