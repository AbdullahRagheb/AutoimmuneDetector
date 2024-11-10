# **Autoimmune Disorder Multi-Label Classification Model**
<a target="_blank" href="https://colab.research.google.com/github/AbdullahRagheb/AutoimmuneDetector/blob/main/multi_label_autoimmune_disorder_Classification.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 


## **Overview**
This repository contains the code and methodology for building and training a multi-label neural network model to classify various autoimmune disorders from clinical data. The model is implemented using PyTorch and includes data preprocessing, model architecture, training, and evaluation.

## **Project Structure**
Here is an overview of the main files in the project:

- **Complete_Updated_Autoimmune_Disorder_Dataset2.csv**: The dataset file containing clinical records.
- **data_preprocessing.py**: Script for data cleaning, encoding, and standardization.
- **dataset.py**: Contains the `AutoimmuneDataset` class for handling the dataset and DataLoader preparation.
- **main.py**: The main script that integrates all components and runs the training and evaluation pipeline.
- **model.py**: Defines the `MultiLabelNN` model architecture.
- **multi_label_nn_model.pth**: Saved weights of the trained model.
- **predict.py**: Script for generating predictions using the trained model.
- **requirements.txt**: List of required dependencies.
- **train.py**: Contains the functions `train_epoch()` and `evaluate_epoch()` for training and validating the model.

## **Dataset**
The dataset used for this project is sourced from the [Kaggle Autoimmune Disorder Dataset](https://www.kaggle.com/datasets) and contains clinical features for over 10,000 records. Key columns include:

- **Age**
- **Gender**
- **Clinical measurements (e.g., RBC count, Hemoglobin, etc.)**
- **Diagnosis**

### **Data Preprocessing**
1. **Handling Categorical Columns**: The `Gender` and `Diagnosis` columns are encoded using `LabelEncoder` and `MultiLabelBinarizer` respectively.
2. **Normalization**: All numerical columns are standardized using `StandardScaler`.
3. **Feature Engineering**: The `Diagnosis` column is transformed into a binary matrix for multi-label classification.

## **Model Architecture**
The neural network architecture used for this project is a fully connected feed-forward model with the following specifications:

- **Input Layer**: Number of features based on preprocessed data.
- **Hidden Layers**:
  - **First layer**: 128 neurons, ReLU activation.
  - **Second layer**: 64 neurons, ReLU activation.
- **Output Layer**: Number of classes equal to the number of unique diagnoses, with softmax activation.

### **Loss Function and Optimization**
- **Loss Function**: Binary Cross-Entropy Loss (`BCELoss`).
- **Optimizer**: Adam optimizer with a learning rate of `0.00001`.
- **Device**: Training on CUDA if available, otherwise on CPU.

## **Training and Evaluation**
The model is trained over 200 epochs with batch size 16. Training includes validation after each epoch to monitor performance metrics such as:

- **Validation Loss**
- **F1 Score (micro and macro)**
- **Accuracy**

### **Key Functions**
- **`train_epoch()`**: Trains the model for one epoch and returns the average training loss.
- **`evaluate_epoch()`**: Evaluates the model on the validation set, calculates metrics, and returns the validation loss, F1 scores, and accuracy.
- **`get_predictions()`**: Generates predictions and converts them into interpretable disease labels using a label mapping.

## **Results**
Metrics from training and validation, including loss and F1 scores, are printed for each epoch. The final evaluation includes generating predictions and mapping them to human-readable disease names.


## **Installation and Setup**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/autoimmune-disorder-classifier.git

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt

3. **Run the main training script**:
   ```bash
   python main.py

