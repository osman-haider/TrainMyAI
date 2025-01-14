import keras
import numpy as np
import io
import matplotlib.pyplot as plt
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense
from tf_keras.optimizers import RMSprop
import os
import seaborn as sns
from PIL import Image


class TextGenModel:
    def __init__(self):
        self.folder_path = "extracted_folder"
        self.file_name = [f for f in os.listdir(self.folder_path) if f.endswith('.txt')][0]
        self.filepath = os.path.join(self.folder_path, self.file_name)
        print(f"self.filepath: {self.filepath}")
        self.char_step = 3
        self.sequence_length = 40
        self.batch_size = 18
        self.learning_rate = 0.01
        self.text = None
        self.chars = None
        self.char_indices = None
        self.indices_char = None
        self.model = None
        self.x = None
        self.y = None
        self.history = None

    def load_data(self):

        with io.open(self.filepath, encoding='utf-8') as f:
            self.text = f.read().lower()
        self.chars = sorted(list(set(self.text)))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}

    def preprocess_data(self):
        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - self.sequence_length, self.char_step):
            sentences.append(self.text[i: i + self.sequence_length])
            next_chars.append(self.text[i + self.sequence_length])
        x = np.zeros((len(sentences), self.sequence_length, len(self.chars)), dtype=bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = True
            y[i, self.char_indices[next_chars[i]]] = True
        self.x = x
        self.y = y

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.sequence_length, len(self.chars))))
        self.model.add(Dense(len(self.chars), activation='softmax'))
        optimizer = RMSprop(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def train_model(self, epochs):
        self.history = self.model.fit(self.x, self.y, batch_size=self.batch_size, epochs=epochs)

    def plot_losses(self):
        plt.plot(self.history.history['loss'])
        plt.title('Model Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper right')
        sns.despine()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def predict_next_chars(self, initial_text, num_chars=20):
        text = initial_text[-self.sequence_length:]
        predictions = []
        for j in range(num_chars):
            x_pred = np.zeros((1, self.sequence_length, len(self.char_indices)), dtype=bool)
            for t, char in enumerate(text):
                if char in self.char_indices:
                    x_pred[0, t, self.char_indices[char]] = True
                else:
                    predictions.append(f"Character '{char}' not in training set.")
                    return predictions
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds)
            next_char = self.indices_char[next_index]
            text += next_char
            text = text[1:]
            predictions.append(f"Prediction after {j + 1} character(s): {initial_text + text[-(j + 1):]}")
        return predictions

    @staticmethod
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)