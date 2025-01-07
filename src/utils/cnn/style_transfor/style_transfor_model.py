import os
import tensorflow as tf
import numpy as np
from tf_keras.applications import VGG19
from tf_keras.models import Model
from tf_keras.preprocessing.image import load_img, img_to_array
from tf_keras.optimizers import Adam
import matplotlib.pyplot as plt

class StyleTransformer:
    def __init__(self, target_size=(256, 256), learning_rate=0.001, alpha=1.0, beta=1.0):
        self.content_dir = "extracted_folder/ContentImages/"
        self.style_dir = "extracted_folder/StyleImages/"
        self.target_size = target_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.vgg_model = self.build_vgg_model()
        self.transformer = self.build_transformer()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.content_losses = []
        self.style_losses = []

    def build_vgg_model(self):
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        output_layers = content_layers + style_layers
        outputs = [vgg.get_layer(name).output for name in output_layers]
        return Model(inputs=vgg.input, outputs=outputs)

    def build_transformer(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(64, (9, 9), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def load_and_preprocess_image(self, image_path):
        img = load_img(image_path, target_size=self.target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0
        return img

    def prepare_dataset(self):
        content_images = [os.path.join(self.content_dir, fname) for fname in os.listdir(self.content_dir)]
        style_images = [os.path.join(self.style_dir, fname) for fname in os.listdir(self.style_dir)]
        return content_images, style_images

    def compute_content_loss(self, content_features, generated_features):
      # Use the first layer (e.g., 'block5_conv2') for content loss
      return tf.reduce_mean(tf.square(content_features[0] - generated_features[0]))


    def gram_matrix(self, tensor):
        channels = int(tensor.shape[-1])
        vectorized = tf.reshape(tensor, [-1, channels])
        gram = tf.matmul(vectorized, vectorized, transpose_a=True)
        return gram / tf.cast(tf.size(tensor), tf.float32)

    def compute_style_loss(self, style_features, generated_features):
      loss = 0
      for sf, gf in zip(style_features, generated_features):
          loss += tf.reduce_mean(tf.square(self.gram_matrix(sf) - self.gram_matrix(gf)))
      return loss / len(style_features)


    def add_noise(self, image, noise_factor=0.1):
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_factor)
        return tf.clip_by_value(image + noise, 0.0, 1.0)

    def train(self, epochs):
        content_images, style_images = self.prepare_dataset()
        for epoch in range(epochs):
            for content_path, style_path in zip(content_images, style_images):
                content_image = self.load_and_preprocess_image(content_path)
                style_image = self.load_and_preprocess_image(style_path)

                with tf.GradientTape() as tape:
                    generated_image = self.transformer(content_image)
                    noisy_generated_image = self.add_noise(generated_image, noise_factor=0.1)

                    content_features = self.vgg_model(content_image)[:1]
                    style_features = self.vgg_model(style_image)[1:]
                    generated_content_features = self.vgg_model(noisy_generated_image)[:1]
                    generated_style_features = self.vgg_model(noisy_generated_image)[1:]

                    content_loss = self.compute_content_loss(content_features, generated_content_features)
                    style_loss = self.compute_style_loss(style_features, generated_style_features)
                    total_loss = self.alpha * content_loss + self.beta * style_loss

                gradients = tape.gradient(total_loss, self.transformer.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

                self.content_losses.append(content_loss.numpy())
                self.style_losses.append(style_loss.numpy())

            print(f"Epoch {epoch + 1}, Content Loss: {content_loss.numpy()}, Style Loss: {style_loss.numpy()}")

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.content_losses, label="Content Loss")
        ax.plot(self.style_losses, label="Style Loss")
        ax.set_title("Training Losses")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        return fig

    def stylize_image(self, content_image_path, style_image_path):
        content_image = self.load_and_preprocess_image(content_image_path)
        style_image = self.load_and_preprocess_image(style_image_path)

        content_transformed = self.transformer(content_image)
        style_transformed = self.transformer(style_image)
        generated_image = (content_transformed + style_transformed) / 2

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(tf.squeeze(content_image))
        axs[0].set_title("Content Image")
        axs[0].axis('off')

        axs[1].imshow(tf.squeeze(style_image))
        axs[1].set_title("Style Image")
        axs[1].axis('off')

        axs[2].imshow(tf.squeeze(generated_image))
        axs[2].set_title("Stylized Image")
        axs[2].axis('off')

        return fig