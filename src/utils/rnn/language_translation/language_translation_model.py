import numpy as np
import pandas as pd
from keras.src.legacy.preprocessing.text import Tokenizer
from tf_keras.models import Sequential
from tf_keras.layers import GRU, Dense, TimeDistributed, Dropout, Embedding
from tf_keras.optimizers import Adam
from tf_keras.losses import sparse_categorical_crossentropy
from tf_keras.utils import pad_sequences
import os

class LanguageTranslationModel:
    # Initialize the model, tokenizers, and start the data processing and model building
    def __init__(self):
        self.folder_path = "extracted_folder"
        self.file_name = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')][0]
        self.data_path = os.path.join(self.folder_path, self.file_name)
        self.model = None
        self.Language_1_tokenizer = None
        self.Language_2_tokenizer = None
        self.input_shape = None
        self.Language_1_vocab_size = None
        self.Language_2_vocab_size = None
        self.preprocess_x = None
        self.preprocess_y = None
        self.history = None
        self.load_and_preprocess_data()
        self.build_model()

    # Load data from CSV and preprocess it
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        self.Language_1_sentences = df['Language 1'].values
        self.Language_2_sentences = df['Language 2'].values
        self.preprocess()

    # Tokenize text data and prepare it for training
    def preprocess(self):
        x_tk, y_tk = Tokenizer(), Tokenizer()
        x_tk.fit_on_texts(self.Language_1_sentences)
        y_tk.fit_on_texts(self.Language_2_sentences)
        self.Language_1_tokenizer, self.Language_2_tokenizer = x_tk, y_tk
        self.preprocess_x = self.pad(x_tk.texts_to_sequences(self.Language_1_sentences))
        self.preprocess_y = self.pad(y_tk.texts_to_sequences(self.Language_2_sentences),
                                     length=max(len(seq) for seq in self.preprocess_x))
        self.preprocess_y = self.preprocess_y.reshape(*self.preprocess_y.shape, 1)
        self.update_vocab_sizes()

    # Update vocabulary sizes and input shape for the model configuration
    def update_vocab_sizes(self):
        self.Language_1_vocab_size = len(self.Language_1_tokenizer.word_index) + 1
        self.Language_2_vocab_size = len(self.Language_2_tokenizer.word_index) + 1
        self.input_shape = (None, self.preprocess_x.shape[-1])

    # Pad sequences to ensure uniform input size
    def pad(self, sequences, length=None):
        return pad_sequences(sequences, maxlen=length, padding='post')

    # Build a neural network model for language translation
    def build_model(self):
        learning_rate = 0.005
        self.model = Sequential([
            Embedding(self.Language_1_vocab_size, 256, input_length=self.input_shape[1]),
            GRU(256, return_sequences=True),
            TimeDistributed(Dense(1024, activation='relu')),
            Dropout(0.5),
            TimeDistributed(Dense(self.Language_2_vocab_size, activation='softmax'))
        ])
        self.model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])

    # Train the model with the given dataset
    def train(self, epochs=10, batch_size=1024, validation_split=0.2):
        self.history = self.model.fit(self.preprocess_x, self.preprocess_y, batch_size=batch_size, epochs=epochs,
                                      validation_split=validation_split)

    # Predict translation of input text
    def predict(self, text):
        sequence = [self.Language_1_tokenizer.word_index.get(word, 0) for word in text.split()]
        padded_sequence = self.pad([sequence], length=self.input_shape[1])
        prediction = self.model.predict(padded_sequence)
        return ' '.join([self.Language_2_tokenizer.index_word.get(np.argmax(word), '') for word in prediction[0]])