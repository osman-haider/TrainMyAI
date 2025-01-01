import numpy as np
import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Binary_Classification:
    def __init__(self):
        """
        Initialize the Binary Classification model and dataset.
        Dynamically retrieves class names from the folder structure.
        """
        self.image_exts = ['jpeg', 'jpg', 'bmp', 'png']
        self.data_dir = 'extracted_folder'
        self.data = tf.keras.utils.image_dataset_from_directory(self.data_dir)
        self.class_names = self.data.class_names  # Retrieve class names dynamically
        self.train = None
        self.test = None
        self.val = None
        self.model = Sequential()
        self.history = None

    def dataset_preprocessing(self):
        """
        Normalize images and prepare the dataset for training.
        """
        self.data = self.data.map(lambda x, y: (x / 255.0, y))
        self.data = self.data.shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    def splitting_dataset(self):
        """
        Properly split the dataset into training, validation, and test sets.
        """
        dataset_size = len(list(self.data))
        train_size = int(0.7 * dataset_size)
        val_size = int(0.2 * dataset_size)

        self.train = self.data.take(train_size)
        self.val = self.data.skip(train_size).take(val_size)
        self.test = self.data.skip(train_size + val_size)

    def model_creation(self):
        """
        Creates and compiles a convolutional neural network (CNN) model.
        """
        self.model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D())

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    def train_model(self, epochs=20, callbacks=None):
        """
        Train the model on the training dataset and validate using the validation dataset.
        """
        if not self.model._is_compiled:
            self.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        self.history = self.model.fit(self.train, epochs=epochs, validation_data=self.val, callbacks=callbacks)
        return self.history

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.history.history['loss'], color='teal', label='loss')
        ax.plot(self.history.history['val_loss'], color='orange', label='val_loss')
        ax.set_title('Loss', fontsize=20)
        ax.legend(loc="upper left")
        return fig

    def plot_accuracy(self):
        fig, ax = plt.subplots()
        ax.plot(self.history.history['accuracy'], color='teal', label='accuracy')
        ax.plot(self.history.history['val_accuracy'], color='orange', label='val_accuracy')
        ax.set_title('Accuracy', fontsize=20)
        ax.legend(loc="upper left")
        return fig

    def inference(self, img):
        """
        Perform inference on a given image tensor and return the prediction result.
        Args:
            img: TensorFlow tensor of the image.
        Returns:
            str: Prediction result based on the folder names.
        """
        # Resize and normalize the image
        resize = tf.image.resize(img, (256, 256)) / 255.0

        # Expand dimensions to match the model input
        yhat = self.model.predict(np.expand_dims(resize.numpy(), axis=0))[0][0]

        # Use dynamically retrieved class names
        predicted_class = self.class_names[int(yhat > 0.5)]  # 0 or 1 -> corresponding class name
        return predicted_class

