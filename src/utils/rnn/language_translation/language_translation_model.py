import numpy as np
import pandas as pd
import collections
from tf_keras.models import Sequential
from tf_keras.layers import GRU, Dense, TimeDistributed, Activation, Dropout, Embedding
from tf_keras.optimizers import Adam
from tf_keras.losses import sparse_categorical_crossentropy
from keras.src.legacy.preprocessing.text import Tokenizer
from tf_keras.utils import pad_sequences
import matplotlib.pyplot as plt
import os
import io
import seaborn as sns
from PIL import Image


class LanguageTranslationModel:
    def __init__(self):
        self.folder_path = "extracted_folder"
        self.file_name = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')][0]
        self.df = pd.read_csv(os.path.join(self.folder_path, self.file_name))
        self.model = None
        self.Language_1_tokenizer = None
        self.Language_2_tokenizer = None
        self.input_shape = None
        self.Language_1_vocab_size = None
        self.Language_2_vocab_size = None
        self.Language_1_sentences = None
        self.Language_2_sentences = None
        self.history = None

    def load_data(self):
        df = self.df
        Language_1_sentences = df['Language 1'].values
        Language_2_sentences = df['Language 2'].values
        self.Language_1_sentences = Language_1_sentences
        self.Language_2_sentences = Language_2_sentences

    def tokenize(self, x):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)
        return tokenizer.texts_to_sequences(x), tokenizer

    def pad(self, x, length=None):
        return pad_sequences(x, maxlen=length, padding='post')

    def preprocess(self):
        self.load_data()
        preprocess_x, x_tk = self.tokenize(self.Language_1_sentences)
        preprocess_y, y_tk = self.tokenize(self.Language_2_sentences)
        preprocess_x = self.pad(preprocess_x)
        preprocess_y = self.pad(preprocess_y)
        preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
        self.input_shape = (None, preprocess_x.shape[-1])
        self.Language_1_vocab_size = len(x_tk.word_index) + 1
        self.Language_2_vocab_size = len(y_tk.word_index) + 1

        return preprocess_x, preprocess_y, x_tk, y_tk

    def build_model(self):
        learning_rate = 0.005
        model = Sequential()
        model.add(Embedding(self.Language_1_vocab_size, 256, input_length=self.input_shape[1]))
        model.add(GRU(256, return_sequences=True))
        model.add(TimeDistributed(Dense(1024, activation='relu')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.Language_2_vocab_size, activation='softmax')))
        model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])
        return model

    def train(self, epochs=10):
        history = self.model.fit(self.Language_1_sentences, self.Language_2_sentences, batch_size=1024, epochs=epochs, validation_split=0.2)
        self.history = history

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.title('Training and Validation Loss Over Time')
        sns.despine()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy Over Time')
        sns.despine()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def predict(self, text):
        y_id_to_word = {value: key for key, value in self.Language_2_tokenizer.word_index.items()}
        y_id_to_word[0] = ''

        # Using .get to safely access the word index dictionary and default to 0 if the word is not found
        sentence = [self.Language_1_tokenizer.word_index.get(word, 0) for word in text.split()]
        sentence = self.pad([sentence], length=self.input_shape[1])

        prediction = self.model.predict(sentence)
        # Constructing the output sentence
        return ' '.join([y_id_to_word[np.argmax(word)] for word in prediction[0]])