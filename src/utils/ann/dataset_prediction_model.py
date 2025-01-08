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

    def get_data_head(self):
        return self.data.head()

    def preprocess(self, drop_columns=None, target_column=None):
        if drop_columns:
            self.data.drop(columns=drop_columns, inplace=True)
        self.data.fillna("UNKNOWN", inplace=True)
        self.data.drop_duplicates(inplace=True)

        for col in self.data.select_dtypes(include=['number']).columns:
            if col != target_column:  # Exclude target column from outlier removal
                self.remove_outliers(col)

        self.encode_categorical(exclude_columns=[target_column])

    def remove_outliers(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]

    def encode_categorical(self, exclude_columns=None):
        exclude_columns = exclude_columns or []
        for col in self.data.select_dtypes(include=['object']).columns:
            if col not in exclude_columns:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])
                self.encoders[col] = encoder
                self.label_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    def split_data(self, predicted_column):
        self.task_type = 'regression' if self.data[predicted_column].dtype in ['int64', 'float64'] else 'classification'

        self.X = self.data.drop(predicted_column, axis=1)
        self.y = self.data[predicted_column]

        if self.task_type == 'classification':
            # Save the LabelEncoder for the target column
            self.target_encoder = LabelEncoder()
            self.y = self.target_encoder.fit_transform(self.y)

        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(19, activation='relu', input_shape=(self.X_train.shape[1],)))  # Specify input shape
        for _ in range(3):
            self.model.add(Dense(19, activation='relu'))

        if self.task_type == 'regression':
            self.model.add(Dense(1))  # Single output for regression
            self.model.compile(optimizer='adam', loss='mse')
        elif self.task_type == 'classification':
            # num_classes = len(np.unique(self.y_train))  # Get the number of unique classes
            num_classes = len(self.target_encoder.classes_)  # Fix: Get the number of classes from the encoder
            self.model.add(Dense(num_classes, activation='softmax'))  # Multi-class output
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=100, batch_size=128):
        self.history = self.model.fit(
            x=self.X_train,
            y=self.y_train,
            validation_data=(self.X_test, self.y_test),
            batch_size=batch_size,
            epochs=epochs
        )

    def evaluate_model(self):
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

    def make_prediction(self, input_data):
        input_df = pd.DataFrame([input_data], columns=self.X.columns)

        # Encode categorical columns using stored encoders
        for col in input_df.select_dtypes(include=['object']).columns:
            if col in self.encoders:
                encoder = self.encoders[col]
                input_df[col] = encoder.transform(input_df[col])
            else:
                raise ValueError(f"Unexpected column '{col}' in input data.")

        # Scale the input data
        scaled_input = self.scaler.transform(input_df)

        # Predict using the model
        prediction = self.model.predict(scaled_input)

        if self.task_type == 'classification':
            # Get the class index
            class_index = np.argmax(prediction, axis=1)[0]
            # Map the index back to the original string label
            original_label = self.target_encoder.inverse_transform([class_index])[0]
            return original_label

        return prediction[0][0]  # For regression, return the numerical prediction