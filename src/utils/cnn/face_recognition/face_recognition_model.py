from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.models import Sequential
from tf_keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
import numpy as np
import tensorflow as tf


class face_recognition:
    """
    This class implements a face recognition system using a Convolutional Neural Network (CNN).
    It provides methods for data preprocessing, model building, training, and inference.
    """
    def __init__(self):
        """
        Initializes the face_recognition class, setting up necessary paths and parameters.
        """
        self.TrainingImagePath = "extracted_folder"
        self.training_set = None
        self.test_set = None
        self.classifier = Sequential()
        self.OutputNeurons = None
        self.history = None

    def train_data_generator(self):
        """
        Prepares the training data generator with data augmentation techniques.
        """
        train_datagen = ImageDataGenerator(
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)

        self.training_set = train_datagen.flow_from_directory(
            self.TrainingImagePath,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')

    def test_data_generator(self):
        """
        Prepares the testing/validation data generator without augmentation.
        """
        test_datagen = ImageDataGenerator()
        self.test_set = test_datagen.flow_from_directory(
            self.TrainingImagePath,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')

    def lookup_table(self):
        """
        Generates a lookup table mapping numeric class indices to class labels.
        """
        TrainClasses = self.training_set.class_indices
        ResultMap = {}
        for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
            ResultMap[faceValue] = faceName

        self.OutputNeurons = len(ResultMap)

    def model_building(self):
        """
        Constructs the CNN model architecture with convolutional, pooling, and fully connected layers.
        """
        self.lookup_table()

        self.classifier.add(
            Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))
        self.classifier.add(MaxPool2D(pool_size=(2, 2)))

        self.classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        self.classifier.add(MaxPool2D(pool_size=(2, 2)))

        self.classifier.add(Flatten())
        self.classifier.add(Dense(64, activation='relu'))
        self.classifier.add(Dense(self.OutputNeurons, activation='softmax'))

    def model_complie(self):
        """
        Compiles the CNN model with categorical cross-entropy loss and Adam optimizer.
        """
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    def model_fit(self, epochs):
        """
        Trains the CNN model using the prepared training and validation data generators.

        Args:
            epochs (int): Number of epochs to train the model.

        Returns:
            History object: Contains details about the training process.
        """
        num_training_samples = len(self.training_set.filenames)
        num_validation_samples = len(self.test_set.filenames)
        steps_per_epoch = num_training_samples // self.training_set.batch_size
        validation_steps = num_validation_samples // self.test_set.batch_size

        self.model_complie()
        self.history = self.classifier.fit(
            self.training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.test_set,
            validation_steps=validation_steps,
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
        img_resized = tf.image.resize(img_tensor, (64, 64))
        img_array = img_resized.numpy() / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.classifier.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)

        class_indices = self.training_set.class_indices
        class_labels = {v: k for k, v in class_indices.items()}
        predicted_label = class_labels[predicted_class[0]]

        return predicted_label