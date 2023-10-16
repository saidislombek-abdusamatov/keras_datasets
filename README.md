# Keras Datasets Image Classification and Boston Housing Price Prediction with Streamlit

This project demonstrates the use of pre-trained Keras models for image classification on popular datasets like MNIST, CIFAR-10, CIFAR-100, and Fashion MNIST. Additionally, it includes a Boston Housing Price Prediction feature using a neural network. The application utilizes Streamlit, a powerful Python library for creating web applications, to provide a user-friendly interface for both image classification and housing price prediction.

## Models Used

- **MNIST**: Handwritten digit classification (0-9).
- **CIFAR-10**: Object classification in 10 classes (e.g., airplanes, cars, birds, etc.).
- **CIFAR-100**: Object classification in 100 fine-grained classes.
- **Fashion MNIST**: Fashion item classification (e.g., shirts, shoes, etc.).
- **Boston Housing Price Prediction**: Predicts housing prices based on various features in the Boston area.

## How to Use

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/saidislombek-abdusamatov/keras_datasets.git
   ```

2. **Install Dependencies:**

   ```bash
   pip install tensorflow streamlit opencv-python numpy
   ```

3. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

   This command will start the Streamlit web application locally. You can access the app in your browser at `http://localhost:8501`.

## Features

### Image Classification:

1. **Upload Image:**
   - Click on the "Upload Image" button to select an image (PNG, JPG, or JPEG format).
   - The uploaded image will be displayed below the button.

2. **Select Model:**
   - Choose the model you want to use from the dropdown menu: MNIST, CIFAR-10, CIFAR-100, or Fashion MNIST.

3. **Prediction:**
   - Click the "Predict!" button to see the model's prediction based on the uploaded image.
   - The predicted class label will be displayed along with a bar chart showing the model's confidence scores for each class.

### Boston Housing Price Prediction:

1. **Dataset Input:**
   - The Boston Housing Price Prediction feature utilizes a neural network trained on the Boston Housing dataset.
   - The input fields for various features (e.g., number of rooms, crime rate, etc.) are provided.

2. **Prediction:**
   - After entering the required values, click the "Predict Price" button to get the predicted housing price based on the input features.

## Models and Classes

- **MNIST Classes:** 0-9 (Digits)
- **CIFAR-10 Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **CIFAR-100 Classes:** Various fine-grained categories
- **Fashion MNIST Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
