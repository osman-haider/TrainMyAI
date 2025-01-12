# TrainMyAI

TrainMyAI is a user-friendly tool designed to enable anyone to train deep learning models with just a single click. Built with a simple and interactive interface powered by Streamlit, this project aims to simplify the process of model training, evaluation, and inference.

## Features
- **Model Selection:** Choose from a variety of deep learning models.
- **Dataset Upload:** Upload your dataset to train the selected model.
- **Model Training:** Automatically trains the model and displays performance metrics.
- **Download Model:** Download the trained model for future use.
- **Model Inference:** Perform inference using the trained model.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.7 or above
- pip (Python package installer)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/osman-haider/TrainMyAI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd TrainMyAI
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Application
1. Open a terminal in the project directory.
2. Run the following command:
   ```bash
   streamlit run main.py
   ```
3. Open the displayed URL in your browser to access the application.

## Supported Models
TrainMyAI supports the following deep learning models and tasks:

1. **Convolutional Neural Networks (CNN):**
   - Image Classification:
     - Binary Classification
     - Multi-Class Classification
   - Object Detection
   - Face Recognition
   - Style Transfer

2. **Artificial Neural Networks (ANN):**
   - Dataset Prediction

3. **Recurrent Neural Networks (RNN):**
   - Sentiment Analysis
   - Time Series Forecasting
   - Text Generation
   - Machine Translation

4. **Long Short-Term Memory Networks (LSTM):**
   - Text Summarization
   - Language Modeling
   - Time Series Forecasting
   - Dataset Prediction

5. **Generative Adversarial Networks (GAN):**
   - Image Generation
   - Image Super-Resolution
   - Facial Image Generation
   - Text-to-Image Generation

## Dataset Requirements
### For CNN Models:
Provide your dataset as a zip file with the following structure:
- For Binary Classification:
  - Zip the folders directly without nesting them under a parent folder. For example:
    ```
    cat/
    dog/
    ```

- For Multiclass Classification:
  - Zip the folders directly without nesting them under a parent folder. For example:
    ```
    class_A/
    class_B/
    class_C/
    class_D/
    ```

- For Object Detection:
  - You must have an `.xml` file or `.csv` file. If you have a `.csv` file, it must be in the root directory along with a second directory named `images`.
  - If you have `.xml` files, the root directory should contain two directories: `images` and `annotations` (where all `.xml` files are stored).
  - Ensure your `.xml` or `.csv` file includes the following columns: `filename` (stores the filenames present in your `images` directory), `xmin`, `ymin`, `xmax`, `ymax`.
  - The dataset must be provided in a `.zip` file.
  - CSV file. For example:
    ```
    images/
      img1.jpg
    {datasetname}.csv
    ```
  - XML. For example:
    ```
    images/
      img1.jpg
    annotations/
      img1.xml
    ```
- For Style Transformer
  - Zip the folders directly without nesting them under a parent folder. For example:
  ```
    ContentImages/
      img1.jpg
    StyleImages/
      img1.jpg
    ```
### For ANN Models:
Provide your dataset as a zip file with the following structure:
- Dataset Prediction:
  - Zip the folders directly without nesting them under a parent folder. For example:
    ```
    dataset_name.csv
    ```
### For RNN Models:
Provide your dataset as a zip file with the following structure:
- Sentiment Analysis:
  - Zip the folders directly without nesting them under a parent folder. For example:
    ```
    dataset_name.csv
    ```
  - Dataset must contain two folders "text" and "label":
    - text: Text could be any language
    - label: positive or negative, 0 or 1
## Project Status
TrainMyAI is a work in progress. Development is ongoing, and new features and updates will be documented in the README file. Your feedback and contributions are welcome!

## Contribution
We warmly welcome contributions to TrainMyAI. Please read the [CONTRIBUTING.md](https://github.com/osman-haider/TrainMyAI/blob/master/CONTRIBUTING.md) file before starting your contributions.

## Contact
Feel free to reach out:
- **Portfolio:** [https://osman-haider.github.io/osman/](https://osman-haider.github.io/osman/)
- **LinkedIn:** [https://www.linkedin.com/in/m-usman-haider/](https://www.linkedin.com/in/m-usman-haider/)
- **Email:** osmanhaider167@gmail.com