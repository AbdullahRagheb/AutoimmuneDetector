import torch
from torch.utils.data import DataLoader
from data_preprocessing import preprocess_data
from dataset import AutoimmuneDataset
from model import MultiLabelNN
from train import train_epoch, evaluate_epoch
from predict import get_predictions, decode_predictions
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# Load and preprocess data
data = pd.read_csv('/Users/ye/Downloads/autoImunneTrasformars/Complete_Updated_Autoimmune_Disorder_Dataset2.csv')  # Load your dataset here
X_train, X_val, y_train, y_val, disease_classes = preprocess_data(data)

# Initialize dataset and DataLoader
train_dataset = AutoimmuneDataset(X_train, y_train)
val_dataset = AutoimmuneDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Set up model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLabelNN(input_dim=X_train.shape[1], n_classes=len(disease_classes)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Train and evaluate the model
num_epochs = 200  # Adjust as necessary

# Initialize lists to store metrics
train_losses = []
val_losses = []
val_f1_scores = []
val_accuracies = []
all_labels = []
all_preds = []

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = evaluate_epoch(model, val_loader, criterion, device)

    # Append metrics to lists
    train_losses.append(train_loss)
    val_losses.append(val_metrics["val_loss"])
    val_f1_scores.append(val_metrics["f1_micro"])
    val_accuracies.append(val_metrics["accuracy"])

    # Store the all_labels and all_preds for confusion matrix after training
    all_labels = val_metrics["all_labels"]
    all_preds = val_metrics["all_preds"]

    # Print metrics for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val F1 Score: {val_metrics['f1_micro']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")


# Make predictions
predictions = get_predictions(model, val_loader, device)
decoded_predictions = decode_predictions(predictions, disease_classes)
for i, labels in enumerate(decoded_predictions):
    print(f"Sample {i+1}: {', '.join(labels)}")
