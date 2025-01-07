import os
import tensorflow as tf
import numpy as np
from tf_keras.applications import VGG19
from tf_keras.models import Model
from tf_keras.preprocessing.image import load_img, img_to_array
from tf_keras.optimizers import Adam
import matplotlib.pyplot as plt

class style_transfor:
    def __init__(self):
        self.content_dir = '/extracted_folder/ContentImages'
        self.style_dir = '/extracted_folder/StyleImages'
        self.transformer = None
        # Training Loop
        self.optimizer = Adam(learning_rate=0.001)
        self.epochs = 10
        self.batch_size = 1
        self.alpha = 1.0  # Content weight
        self.beta = 1.0  # Style weight


    def load_and_preprocess_image(self, image_path, target_size=(256, 256)):
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize to [0, 1]
        return img

    def prepare_dataset(self):

        content_images = [os.path.join(self.content_dir, fname) for fname in os.listdir(self.content_dir)]
        style_images = [os.path.join(self.style_dir, fname) for fname in os.listdir(self.style_dir)]
        return content_images, style_images

    def build_vgg_model(self):
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in ['block5_conv2', 'block1_conv1']]
        return Model(inputs=vgg.input, outputs=outputs)

    # Loss Functions
    def compute_content_loss(self, content_features, generated_features):
        return tf.reduce_mean(tf.square(content_features - generated_features))

    def gram_matrix(self, tensor):
        channels = int(tensor.shape[-1])
        vectorized = tf.reshape(tensor, [-1, channels])
        gram = tf.matmul(vectorized, vectorized, transpose_a=True)
        return gram / tf.cast(tf.size(tensor), tf.float32)

    def compute_style_loss(self, style_features, generated_features):
        style_gram = self.gram_matrix(style_features)
        generated_gram = self.gram_matrix(generated_features)
        return tf.reduce_mean(tf.square(style_gram - generated_gram))

    def compute_total_loss(self, content_features, style_features, generated_features, alpha=1.0, beta=1.0):
        content_loss = self.compute_content_loss(content_features[0], generated_features[0])
        style_loss = self.compute_style_loss(style_features[1], generated_features[1])
        return alpha * content_loss + beta * style_loss

    def model_creation(self):

        self.transformer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(64, (9, 9), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')  # Output 3 channels
        ])

    def call(self, content_image, style_image):
        vgg = self.build_vgg_model()
        content_features = vgg(content_image)
        style_features = vgg(style_image)
        generated_image = self.transformer(content_image)
        generated_features = vgg(generated_image)
        return content_features, style_features, generated_features

    def traning(self):
        for epoch in range(self.epochs):
            for content_path, style_path in zip(self.content_images, self.style_images):
                content_image = self.load_and_preprocess_image(content_path)
                style_image = self.load_and_preprocess_image(style_path)

                with tf.GradientTape() as tape:
                    content_features, style_features, generated_features = self.transformer(content_image, style_image)
                    loss = self.compute_total_loss(content_features, style_features, generated_features, self.alpha, self.beta)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

        print("Training Complete!")