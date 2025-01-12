import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
from src.utils.rnn.sentiment_analysis import sentiment_rnn
import os
import io
import seaborn as sns
from PIL import Image

class SentimentAnalysis:
    """
    Sentiment Analysis class initializes and manages the processes for loading data,
    creating vocabulary, tokenizing reviews, encoding labels, padding features,
    splitting data, initializing the model, training the model, and plotting losses.
    """
    def __init__(self, seq_length=30, batch_size=50):
        """
        Initializes the SentimentAnalysis class with default sequence length and batch size.
        It sets up the device based on CUDA availability and prepares data loaders and losses tracking.
        """
        self.folder_path = "extracted_folder"
        self.file_name = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')][0]
        self.filepath = os.path.join(self.folder_path, self.file_name)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
        self.vocab_to_int = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []

    def load_data(self):
        """
        Loads and preprocesses data from a CSV file, creating vocabulary,
        tokenizing reviews, encoding labels, and padding features.
        """
        data = pd.read_csv(self.filepath)
        reviews = np.array(data['text'])
        labels = np.array(data['label'])

        all_reviews = 'separator'.join(reviews).lower()
        all_text = ''.join([c for c in all_reviews if c not in self.punctuation])
        reviews_split = all_text.split('separator')
        words = all_text.split()

        self.create_vocab(words)
        reviews_ints = self.tokenize_reviews(reviews_split)
        encoded_labels = self.encode_labels(labels)
        features = self.pad_features(reviews_ints)

        return self.split_data(features, encoded_labels)

    def create_vocab(self, words):
        """
        Creates a vocabulary from a list of words.
        """
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        self.vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    def tokenize_reviews(self, reviews_split):
        """
        Tokenizes each review into integers using the vocabulary.
        """
        reviews_ints = []
        for review in reviews_split:
            review = review.split()
            reviews_ints.append([self.vocab_to_int[word] for word in review if word in self.vocab_to_int])
        return reviews_ints

    def encode_labels(self, labels):
        """
        Encodes labels as binary values.
        """
        encoded_labels = [1 if label == 'neutral' else 0 if label == 'negative' else 1 for label in labels]
        return np.array(encoded_labels)

    def pad_features(self, reviews_ints):
        """
        Pads or truncates each review to the fixed sequence length.
        """
        features = np.zeros((len(reviews_ints), self.seq_length), dtype=int)
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:self.seq_length]
        return features

    def split_data(self, features, encoded_labels):
        """
        Splits data into training, validation, and test sets.
        """
        split_frac = 0.8
        split_idx = int(len(features) * split_frac)
        train_x, remaining_x = features[:split_idx], features[split_idx:]
        train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

        test_idx = int(len(remaining_x) * 0.5)
        val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

        self.train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), shuffle=True, batch_size=self.batch_size)
        self.valid_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), shuffle=True, batch_size=self.batch_size)
        self.test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)), shuffle=True, batch_size=self.batch_size)

    def init_model(self):
        """
        Initializes the Sentiment RNN model with specific dimensions and dropout probability.
        """
        vocab_size = len(self.vocab_to_int) + 1
        output_size = 1
        embedding_dim = 200
        hidden_dim = 128
        n_layers = 2
        drop_prob = 0.5
        self.net = sentiment_rnn.SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob)
        self.net.to(self.device)

    def train(self, epochs=10, lr=0.001, clip=5):
        """
        Trains the model using specified parameters, applying gradient clipping.
        """
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        for epoch in range(epochs):
            train_losses = []
            val_losses = []

            # Training phase
            self.net.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.net.zero_grad()

                output, _ = self.net(inputs, self.net.init_hidden(inputs.size(0)))
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            self.net.eval()
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output, _ = self.net(inputs, self.net.init_hidden(inputs.size(0)))
                    loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(loss.item())

            # Record the average loss per epoch
            self.train_losses.append(np.mean(train_losses))
            self.val_losses.append(np.mean(val_losses))

            # Print out the losses for the epoch
            print(f"Epoch: {epoch + 1}, Train Loss: {self.train_losses[-1]:.4f}, Validation Loss: {self.val_losses[-1]:.4f}")

    def validate(self, hidden, criterion):
        """
        Validates the model performance on the validation set using a given hidden state and loss criterion.
        """
        val_losses = []
        for inputs, labels in self.valid_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            hidden = tuple([each.data for each in hidden])

            output, hidden = self.net(inputs, hidden)
            val_loss = criterion(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())

        return np.mean(val_losses)

    def plot_losses(self):
        """
        Plots training and validation losses over time.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Number of Batches Processed')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Over Time')
        sns.despine()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def predict(self, test_review):
        """
        Predicts sentiment for a given review using the trained model.
        """
        self.net.eval()
        test_ints = self.tokenize_reviews([test_review])
        features = self.pad_features(test_ints)
        feature_tensor = torch.from_numpy(features).to(self.device)

        batch_size = feature_tensor.size(0)
        h = self.net.init_hidden(batch_size)
        output, h = self.net(feature_tensor, h)

        pred = torch.round(output.squeeze())
        return pred.item()