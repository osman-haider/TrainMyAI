import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class multiclass_classifiction:
    """
    This class handles the entire process of creating, training, and evaluating a CNN model for multi-class image classification.
    """
    def __init__(self):
        """
        Initialize the class with necessary attributes such as data directories, model parameters, and placeholders for generators and models.
        """
        self.data_dir = 'extracted_folder'
        self.train_generator = None
        self.val_generator = None
        self.model = None
        self.history = None

        self.batch_size = 32
        self.image_size = (150, 150)  # Resize all images to this size
        self.num_classes = len(os.listdir(self.data_dir))  # Number of subfolders/classes

    def train_data_generator(self):
        """
        Create and configure the training data generator with data augmentation.
        """
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

        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

    def val_data_generator(self):
        """
        Create and configure the validation data generator without data augmentation.
        """
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
        """
        Build the Convolutional Neural Network (CNN) model with several layers.
        """
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
        """
        Compile the CNN model with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
        """
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def model_fit(self, epochs=20):
        """
        Train the CNN model using the training and validation data generators for a specified number of epochs.

        Args:
            epochs (int): Number of epochs for training the model.

        Returns:
            History object containing details about the training process.
        """
        self.model_complie()
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            verbose=1
        )

        return self.history

    def inference(self, img_tensor):
        """
        Perform inference on a given image tensor and return the prediction result.

        Args:
            img_tensor (tf.Tensor): TensorFlow tensor of the image.

        Returns:
            str: Prediction result based on the folder names.
        """
        img_resized = tf.image.resize(img_tensor, self.image_size)
        img_array = img_resized.numpy() / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        class_indices = self.train_generator.class_indices
        class_labels = {v: k for k, v in class_indices.items()}  # Reverse the dictionary
        predicted_label = class_labels[predicted_class[0]]

        return predicted_label