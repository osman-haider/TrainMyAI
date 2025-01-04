import os
import random
import xml.etree.ElementTree as ET
import csv
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import skimage
from skimage import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from PIL import Image


class object_detection:

    def __init__(self):
        self.DIR_ANNOTATIONS = "extracted_folder/dataset.csv"
        self.DIR_IMAGES = "extracted_folder/images"
        self.model = None
        self.history = None
        self.train_dataframe = None
        self.val_dataframe = None
        self.BUFFER_SIZE = 200
        self.BATCH_SIZE = 10
        self.SIZE = 224
        self.train_steps = None
        self.train_dataset = None
        self.val_steps = None
        self.val_dataset = None
        self.INPUT_SHAPE = (224, 224, 1)
        self.COMPILE_OPTIMIZER = tf.keras.optimizers.Adam()
        self.COMPILE_LOSS = tf.keras.losses.MeanSquaredError()
        self.COMPILE_METRICS = ['accuracy']
        self.FIT_EPOCHS = 100


    def load_normalize_image(self, image_path, img_size, channels=1):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.cast(image, tf.float64)
        image = tf.image.resize(image, size=img_size)
        image = image / 255.0
        return image

    def create_dataset_from_dataframe(self, dataframe):
        image_paths = tf.cast(dataframe.iloc[:]['filename'].values, tf.string)
        image_coordinates = tf.cast(dataframe[['xmin', 'ymin', 'xmax', 'ymax']].values, tf.float64)
        return tf.data.Dataset.from_tensor_slices(tensors=(image_paths, image_coordinates))

    def preprocess_dataset(self):
        dataframe_original = pd.read_csv(self.DIR_ANNOTATIONS)
        dataframe_original['image'] = dataframe_original['image'].apply(lambda x: os.path.join(self.DIR_IMAGES, x))

        dataframe_original = dataframe_original.rename(columns={'image': 'filename'})

        dataframe_preprocessed = dataframe_original.copy()
        dataframe_preprocessed[["xmin", "ymin", "xmax", "ymax"]] /= self.SIZE

        # Split the DataFrame into training(80%) and validation(20%)
        self.train_dataframe, self.val_dataframe = train_test_split(dataframe_preprocessed, test_size=0.2)

    def train_dataset_method(self):
        # Train Dataset
        train_dataset = self.create_dataset_from_dataframe(self.train_dataframe)
        map_func = lambda image_path, data: (self.load_normalize_image(image_path, [224, 224]), data)
        train_dataset = train_dataset.map(map_func=map_func, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_dataset = train_dataset.shuffle(self.BUFFER_SIZE).cache().repeat().batch(self.BATCH_SIZE).prefetch(
            buffer_size=tf.data.AUTOTUNE)
        self.train_steps = max(1, len(self.train_dataframe) // self.BATCH_SIZE)

    def val_dataset_method(self):
        # Test Dataset
        val_dataset = self.create_dataset_from_dataframe(self.val_dataframe)
        map_func = lambda image_path, data: (self.load_normalize_image(image_path, [224, 224]), data)
        val_dataset = val_dataset.map(map_func=map_func, num_parallel_calls=tf.data.AUTOTUNE)
        self.val_dataset = val_dataset.shuffle(self.BUFFER_SIZE).cache().repeat().batch(self.BATCH_SIZE).prefetch(
            buffer_size=tf.data.AUTOTUNE)
        self.val_steps = max(1, len(self.val_dataframe) // self.BATCH_SIZE)

    def call_back(self):
        FIT_CALLBACKS = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=25,
                                                          min_delta=0.001,
                                                          restore_best_weights=True,
                                                          verbose=1)]
        return FIT_CALLBACKS

    def model_creation(self):
        # Model creation
        inputs = tf.keras.Input(shape=self.INPUT_SHAPE, name='input_layer')
        x = tf.keras.layers.Conv2D(8, 3, padding='same', activation=tf.keras.activations.relu, name='conv_layer1')(
            inputs)
        x = tf.keras.layers.MaxPool2D(name='maxpool_layer1')(x)
        x = tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.keras.activations.relu, name='conv_layer2')(x)
        x = tf.keras.layers.MaxPool2D(name='maxpool_layer2')(x)
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.keras.activations.relu, name='conv_layer3')(x)
        x = tf.keras.layers.MaxPool2D(name='maxpool_layer3')(x)
        x = tf.keras.layers.Flatten(name='flatten_layer')(x)
        x = tf.keras.layers.Dense(512, activation=tf.keras.activations.relu, name='dense_layer1')(x)
        x = tf.keras.layers.Dense(32, activation=tf.keras.activations.relu, name='dense_layer2')(x)
        # The last 4 units correspond to the coordinates and size (xmin,ymin,xmax,ymax)
        outputs = tf.keras.layers.Dense(units=4, activation='linear', name='output_layer')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='object_detection')

        # Compilation of the final model
        self.model.compile(
            optimizer=self.COMPILE_OPTIMIZER,
            loss=self.COMPILE_LOSS,
            metrics=self.COMPILE_METRICS)

    def train_model(self, FIT_EPOCHS):
        FIT_CALLBACKS = self.call_back()
        # Model training
        self.history = self.model.fit(
            self.train_dataset,
            steps_per_epoch=self.train_steps,
            validation_data=self.val_dataset,
            validation_steps=self.val_steps,
            batch_size=self.BATCH_SIZE,
            epochs=FIT_EPOCHS,
            callbacks=FIT_CALLBACKS)

    def plot_predictions(self, prediction, image_path, img_size, box_color):
        """
        Plots the prediction result for a single image.

        Args:
            prediction: The predicted bounding box (normalized).
            image_path: Path to the image.
            img_size: Size to scale the prediction (e.g., 224 if [224, 224]).
            box_color: Color of the bounding box.

        Returns:
            The plotted figure.
        """
        # Scale prediction to original image size
        predicted = prediction * img_size

        # Load the image
        image = io.imread(image_path)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)

        # Draw predicted bounding box
        predicted_width = predicted[2] - predicted[0]
        predicted_height = predicted[3] - predicted[1]
        ax.add_patch(patches.Rectangle(
            (predicted[0], predicted[1]),
            predicted_width,
            predicted_height,
            fill=False,
            color=box_color,
            linewidth=2
        ))
        ax.set_title("Predicted Bounding Box", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        return fig

    def inference(self, image_path):
        """
        Perform inference on a given image and return the image with the predicted bounding box.

        Args:
            image_path (str): Path to the input image.

        Returns:
            io.BytesIO: In-memory file containing the image with the predicted bounding box.
        """
        # Preprocess and predict
        image = self.load_normalize_image(image_path, [224, 224])
        expanded_image = tf.expand_dims(image, 0)
        prediction = self.model.predict(expanded_image)

        prediction = np.squeeze(prediction)

        # Plot and save the figure
        fig = self.plot_predictions(
            prediction=prediction,
            image_path=image_path,
            img_size=224,
            box_color='lime'
        )

        import io
        # Convert the matplotlib figure to a PIL Image
        canvas = FigureCanvas(fig)
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        buffer.seek(0)
        image = Image.open(buffer)

        return image