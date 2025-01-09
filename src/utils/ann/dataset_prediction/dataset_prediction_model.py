import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, classification_report
from tf_keras.models import Sequential
from tf_keras.layers import Dense
import json
import io
from PIL import Image
import os

class DatasetPrediction:
    def __init__(self):
        """
        Initializes the DatasetPrediction class, including reading the CSV data file,
        setting up the MinMaxScaler, and initializing variables for the model, history,
        encoders, and task type.
        """
        self.folder_path = "extracted_folder"
        self.file_name = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')][0]
        self.data = pd.read_csv(os.path.join(self.folder_path, self.file_name))
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.y_train, self.y_test = None, None
        self.X_train, self.X_test = None, None
        self.X, self.y = None, None
        self.encoders = {}
        self.label_mappings = {}
        self.task_type = None
        self.target_encoder = None
        self.test_data = None

    def get_data_head(self):
        """
        Returns the first five rows of the dataset.
        """
        return self.data.head()

    def preprocess(self, drop_columns=None, target_column=None):
        """
        Preprocesses the dataset by dropping specified columns, filling missing values,
        removing duplicates, and encoding categorical variables. Also removes outliers
        from numeric columns.
        """
        if drop_columns:
            self.data.drop(columns=drop_columns, inplace=True)
        self.data.fillna("UNKNOWN", inplace=True)
        self.data.drop_duplicates(inplace=True)

        for col in self.data.select_dtypes(include=['number']).columns:
            if col != target_column:
                self.remove_outliers(col)
        self.test_data = self.data.sample(frac=0.05)
        self.encode_categorical(exclude_columns=[target_column])

    def remove_outliers(self, column):
        """
        Removes outliers from a specified numeric column using the IQR method.
        """
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]

    def encode_categorical(self, exclude_columns=None):
        """
        Encodes categorical columns using LabelEncoder, excluding specified columns from encoding.
        """
        exclude_columns = exclude_columns or []
        for col in self.data.select_dtypes(include=['object']).columns:
            if col not in exclude_columns:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])
                self.encoders[col] = encoder
                self.label_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    def split_data(self, predicted_column):
        """
        Splits the dataset into training and testing sets. Determines the task type
        (regression or classification) based on the data type of the predicted column.
        """
        self.task_type = 'regression' if self.data[predicted_column].dtype in ['int64', 'float64'] else 'classification'
        self.X = self.data.drop(predicted_column, axis=1)
        self.y = self.data[predicted_column]

        if self.task_type == 'classification':
            self.target_encoder = LabelEncoder()
            self.y = self.target_encoder.fit_transform(self.y)

        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

    def build_model(self):
        """
        Builds a Sequential neural network model based on the task type. The model
        architecture includes multiple Dense layers with ReLU activation for both
        regression and classification tasks.
        """
        self.model = Sequential()
        self.model.add(Dense(19, activation='relu', input_shape=(self.X_train.shape[1],)))
        for _ in range(3):
            self.model.add(Dense(19, activation='relu'))

        if self.task_type == 'regression':
            self.model.add(Dense(1))
            self.model.compile(optimizer='adam', loss='mse')
        elif self.task_type == 'classification':
            num_classes = len(self.target_encoder.classes_)
            self.model.add(Dense(num_classes, activation='softmax'))
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=100, batch_size=128):
        """
        Trains the model using the provided number of epochs and batch size.
        Tracks the training history for later analysis.
        """
        self.history = self.model.fit(
            x=self.X_train,
            y=self.y_train,
            validation_data=(self.X_test, self.y_test),
            batch_size=batch_size,
            epochs=epochs
        )

    def evaluate_model(self):
        """
        Evaluates the trained model on the test dataset. Returns performance metrics
        based on the task type (MAE, MSE, RMSE, variance score for regression; accuracy
        and classification report for classification).
        """
        predictions = self.model.predict(self.X_test)

        if self.task_type == 'regression':
            metrics = {
                'MAE': mean_absolute_error(self.y_test, predictions),
                'MSE': mean_squared_error(self.y_test, predictions),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, predictions)),
                'Variance Score': explained_variance_score(self.y_test, predictions)
            }
        elif self.task_type == 'classification':
            predictions = np.argmax(predictions, axis=1)
            accuracy = np.mean(predictions == self.y_test)
            metrics = {
                'Accuracy': accuracy,
                'Classification Report': classification_report(self.y_test, predictions, output_dict=True)
            }

        return json.dumps(metrics, indent=4)

    def plot_training_loss(self):
        """
        Plots the training and validation loss over epochs. Returns the plot as an
        image object that can be displayed or saved.
        """
        losses = pd.DataFrame(self.history.history)
        plt.figure(figsize=(15, 5))
        sns.lineplot(data=losses, lw=3)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch')
        sns.despine()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def prediction(self):
        """
        Makes predictions on a small test sample extracted from the dataset. Handles
        missing values and encodes categorical features as done during training. Returns
        the predicted results appended to the test data.
        """
        new_data = self.test_data.copy()
        testing = self.test_data.copy()
        new_data.fillna("UNKNOWN", inplace=True)

        for col in self.encoders:
            encoder = self.encoders[col]
            known_labels = set(encoder.classes_)
            new_data[col] = new_data[col].apply(lambda x: x if x in known_labels else 'UNKNOWN')
            if 'UNKNOWN' not in encoder.classes_:
                extended_classes = np.append(encoder.classes_, 'UNKNOWN')
                encoder.classes_ = extended_classes
            new_data[col] = encoder.transform(new_data[col])

        new_data_features = new_data[self.X.columns]
        new_data_scaled = self.scaler.transform(new_data_features)
        predictions = self.model.predict(new_data_scaled)

        if self.task_type == 'regression':
            predictions = np.round(predictions)
        if self.task_type == 'classification':
            predictions = self.target_encoder.inverse_transform(np.argmax(predictions, axis=1))
        else:
            predictions = predictions.flatten()

        testing['Predicted Result'] = predictions
        return testing