import torch
import numpy as np

def get_predictions(model, data_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            outputs = model(features)
            preds = (outputs > threshold).int().cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)

def decode_predictions(predictions, label_mapping):
    predicted_labels = []
    for sample in predictions:
        diseases = [label_mapping[idx] for idx, label in enumerate(sample) if label == 1]
        predicted_labels.append(diseases if diseases else ["No labels predicted"])
    return predicted_labels
