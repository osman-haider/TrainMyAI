import numpy as np
import io
import matplotlib.pyplot as plt
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, LSTM, Dense
from tf_keras.utils import to_categorical
import os
import seaborn as sns
from PIL import Image

class TextModel:
    def __init__(self):
        self.folder_path = "extracted_folder"
        self.file_name = [f for f in os.listdir(self.folder_path) if f.endswith('.txt')][0]
        self.filepath = os.path.join(self.folder_path, self.file_name)
        self.embedding_dim = 100
        self.lstm_units = 150
        self.epochs = 32
        self.tokenizer = Tokenizer()
        self.model = None
        self.vocab_size = None
        self.max_len = None
        self.history = None
        self.text = None
        self.X = None
        self.y = None

    def read_data(self):
        with io.open(self.filepath, encoding='utf-8') as f:
            text = f.read().lower()
        self.text = text

    def preprocess(self):
        self.tokenizer.fit_on_texts([self.text])
        word_index = self.tokenizer.word_index
        self.vocab_size = len(word_index) + 1
        sequences = []
        for sentence in self.text.split("."):
            tokens = self.tokenizer.texts_to_sequences([sentence])[0]
            for i in range(1, len(tokens)):
                sequences.append(tokens[:i+1])
        self.max_len = max([len(x) for x in sequences])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='pre')
        X = padded_sequences[:, :-1]
        y = to_categorical(padded_sequences[:, -1], num_classes=self.vocab_size)
        self.X = X
        self.y = y

    def build_model(self):
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.X.shape[1]),
            LSTM(self.lstm_units, return_sequences=True),
            LSTM(self.lstm_units),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, batch_size):
        self.history = self.model.fit(self.X, self.y, epochs=self.epochs, batch_size=batch_size)

    def plot_loss(self):
        plt.plot(self.history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        sns.despine()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)


    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        sns.despine()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def predict(self, initial_text, num_words=10):
        text = initial_text.lower()
        results = []  # Initialize a list to store the results
        for _ in range(num_words):
            token_text = self.tokenizer.texts_to_sequences([text])[0]
            padded_token_text = pad_sequences([token_text], maxlen=self.max_len, padding='pre')
            pos = np.argmax(self.model.predict(padded_token_text))
            next_word = [word for word, index in self.tokenizer.word_index.items() if index == pos]
            if next_word:
                text += ' ' + next_word[0]
                results.append(text)  # Append the updated text to the results list
        return results