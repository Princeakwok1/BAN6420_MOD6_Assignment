import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()   ## This loads the Fashion MNIST dataset, splitting it into training and test sets.

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0   ## The pixel values of the images are scaled to be between 0 and 1 for better model performance.

# Reshape the data to fit CNN input
x_train = x_train.reshape(-1, 28, 28, 1)    ## The dataset is reshaped to include a single channel for grayscale images.
x_test = x_test.reshape(-1, 28, 28, 1)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),                   ## A simple is difined with
    MaxPooling2D((2,2)),                                    ## Two convolutional layers 
    Flatten(),                                              ## Two max-pooling layers
    Dense(128, activation='relu'),                          ## A flatten layer and two fully connected layers
    Dense(10, activation='softmax')
])                ## 

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))         ## The model is compiled with the Adam optimizer and trained for 5 epochs.

# Make predictions on two test images
sample_images = x_test[:2]                          ## After training, the model makes predictions for two images from the test set.
predictions = model.predict(sample_images)

# Display predictions
for i in range(2):
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {np.argmax(predictions[i])}')
    plt.show()

# Save the model
model.save("fashion_mnist_cnn.h5")
