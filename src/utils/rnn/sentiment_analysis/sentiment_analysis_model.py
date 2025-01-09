import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
from src.utils.rnn.sentiment_analysis import sentiment_rnn

class SentimentAnalysis:
    def __init__(self, seq_length=30, batch_size=50):
        self.filepath = "extracted_folder/Tweets.csv"
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
        self.vocab_to_int = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = None
        self.val_losses = None

    def load_data(self):
        print(self.filepath)
        data = pd.read_csv(self.filepath)
        reviews = np.array(data['text'])[:14000]
        labels = np.array(data['airline_sentiment'])[:14000]

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
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        self.vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    def tokenize_reviews(self, reviews_split):
        reviews_ints = []
        for review in reviews_split:
            review = review.split()
            reviews_ints.append([self.vocab_to_int[word] for word in review if word in self.vocab_to_int])
        return reviews_ints

    def encode_labels(self, labels):
        encoded_labels = [1 if label == 'neutral' else 0 if label == 'negative' else 1 for label in labels]
        return np.array(encoded_labels)

    def pad_features(self, reviews_ints):
        features = np.zeros((len(reviews_ints), self.seq_length), dtype=int)
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:self.seq_length]
        return features

    def split_data(self, features, encoded_labels):
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
        vocab_size = len(self.vocab_to_int) + 1
        output_size = 1
        embedding_dim = 200
        hidden_dim = 128
        n_layers = 2
        drop_prob = 0.5
        self.net = sentiment_rnn.SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob)
        self.net.to(self.device)

    def train(self, epochs=10):
        lr = 0.001
        clip = 5
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        train_losses = []
        val_losses = []

        self.net.train()
        for epoch in range(epochs):
            h = self.net.init_hidden(self.batch_size)

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                h = tuple([each.data for each in h])

                self.net.zero_grad()
                output, h = self.net(inputs, h)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                optimizer.step()

                train_losses.append(loss.item())

                if len(train_losses) % 100 == 0:
                    self.net.eval()
                    val_h = self.net.init_hidden(self.batch_size)
                    val_loss = self.validate(val_h, criterion)
                    val_losses.append(val_loss)
                    self.net.train()

            print(f"Epoch: {epoch+1}, Train Loss: {np.mean(train_losses):.4f}, Validation Loss: {np.mean(val_losses):.4f}")

        self.train_losses = train_losses
        self.val_losses = val_losses

    def validate(self, hidden, criterion):
        val_losses = []
        for inputs, labels in self.valid_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            hidden = tuple([each.data for each in hidden])

            output, hidden = self.net(inputs, hidden)
            val_loss = criterion(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())

        return np.mean(val_losses)

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Number of Batches Processed')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Over Time')
        plt.show()

    def predict(self, test_review):
        self.net.eval()
        test_ints = self.tokenize_reviews([test_review])
        features = self.pad_features(test_ints)
        feature_tensor = torch.from_numpy(features).to(self.device)

        batch_size = feature_tensor.size(0)
        h = self.net.init_hidden(batch_size)
        output, h = self.net(feature_tensor, h)

        pred = torch.round(output.squeeze())
        return pred.item()


sa = SentimentAnalysis()

# Step 2: Load and preprocess the data
sa.load_data()

# Step 3: Initialize the model
sa.init_model()

# Step 4: Train the model
# You can adjust epochs, learning rate (lr), and clip according to your needs
sa.train(epochs=5, )

test_review = "This product was great!"
predicted_sentiment = sa.predict(test_review)
print(f'Predicted Sentiment: {"Positive" if predicted_sentiment == 1 else "Negative"}')