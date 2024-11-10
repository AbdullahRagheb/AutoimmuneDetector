import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    # Convert 'Diagnosis' column to list of lists
    data['Diagnosis'] = data['Diagnosis'].apply(lambda x: [x])

    # One-hot encode the Diagnosis column
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['Diagnosis'])
    disease_classes = mlb.classes_

    # Convert categorical columns to numerical format
    data = pd.get_dummies(data.drop(columns=['Diagnosis']), drop_first=True)
    
    # Standardize numerical features
    scaler = StandardScaler()
    features = scaler.fit_transform(data)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, disease_classes