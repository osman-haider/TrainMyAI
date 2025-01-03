import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
import numpy as np
from tf_keras.preprocessing import image
import scipy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class multiclass_classifiction:
    def __init__(self):
        self.data_dir = 'extracted_folder'
        # self.data = tf.keras.utils.image_dataset_from_directory(self.data_dir)
        # self.class_names = self.data.class_names  # Retrieve class names dynamically
        self.train_generator = None
        self.val_generator = None
        self.model = None
        self.history = None

        # Parameters
        self.batch_size = 32
        self.image_size = (150, 150)  # Resize all images to this size
        self.num_classes = len(os.listdir(self.data_dir))  # Number of subfolders/classes

    def train_data_generator(self):
        # Create an ImageDataGenerator for training with data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,  # Normalize pixel values
            rotation_range=30,  # Random rotation
            width_shift_range=0.2,  # Random horizontal shift
            height_shift_range=0.2,  # Random vertical shift
            shear_range=0.2,  # Shearing transformations
            zoom_range=0.2,  # Random zoom
            horizontal_flip=True,  # Random horizontal flip
            fill_mode='nearest',  # Filling strategy for empty pixels
            validation_split=0.2  # Reserve 20% for validation
        )

        # Training data generator with augmentation
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

    def val_data_generator(self):
        # Validation data generator (no augmentation, only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

    def cnn_model_buliding(self):
        # Build the CNN model
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

    def model_complie(self):
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def model_fit(self, epochs=20):

        self.model_complie()
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            verbose=1
        )

        return self.history

    def plot_loss(self):
        """
        Plot the training and validation loss over epochs.

        Returns:
            matplotlib.figure.Figure: Figure containing the loss plot.
        """
        fig, ax = plt.subplots()
        ax.plot(self.history.history['loss'], color='teal', label='loss')
        ax.plot(self.history.history['val_loss'], color='orange', label='val_loss')
        ax.set_title('Loss', fontsize=20)
        ax.legend(loc="upper left")
        return fig

    def plot_accuracy(self):
        """
        Plot the training and validation accuracy over epochs.

        Returns:
            matplotlib.figure.Figure: Figure containing the accuracy plot.
        """
        fig, ax = plt.subplots()
        ax.plot(self.history.history['accuracy'], color='teal', label='accuracy')
        ax.plot(self.history.history['val_accuracy'], color='orange', label='val_accuracy')
        ax.set_title('Accuracy', fontsize=20)
        ax.legend(loc="upper left")
        return fig

    def inference(self, img_tensor):
        """
        Perform inference on a given image tensor and return the prediction result.

        Args:
            img_tensor (tf.Tensor): TensorFlow tensor of the image.

        Returns:
            str: Prediction result based on the folder names.
        """
        # Resize the image to match the model's input size
        img_resized = tf.image.resize(img_tensor, self.image_size)
        img_array = img_resized.numpy() / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Decode the predicted class
        class_indices = self.train_generator.class_indices
        class_labels = {v: k for k, v in class_indices.items()}  # Reverse the dictionary
        predicted_label = class_labels[predicted_class[0]]

        return predicted_label

