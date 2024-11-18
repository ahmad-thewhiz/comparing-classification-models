# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from time import time
import argparse
import os
import sys
import pickle
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the Neural Network architecture
class SentimentANN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.5):
        super(SentimentANN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def main(dataset_dir, output_dir, batch_size=1024, epochs=5, learning_rate=1e-3, feature_dim=600, print_every=1000):
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Importing the datasets
    try:
        print("Loading training data...")
        df_train = pd.read_csv(os.path.join(dataset_dir, 'train_data.csv')).sample(n=500000, random_state=42)    
        print(f"Training data loaded with {df_train.shape[0]} samples.")
        
        print("Loading testing data...")
        df_test = pd.read_csv(os.path.join(dataset_dir, 'test_data.csv'))
        print(f"Testing data loaded with {df_test.shape[0]} samples.")
    except FileNotFoundError as e:
        print(f'Error: {e}')
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        print(f'Error: One of the CSV files is empty or malformed: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'An unexpected error occurred while reading the datasets: {e}')
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")

    print(f"\nTraining Dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns")
    print(f"Testing Dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns")

    # Display first 5 rows of training data
    print("\nFirst 5 rows of Training Data:")
    print(df_train.head(5))

    # Check if required columns exist
    required_columns = ['sentence', 'sentiment']
    for col in required_columns:
        if col not in df_train.columns or col not in df_test.columns:
            print(f"Error: '{col}' column not found in one of the datasets.")
            sys.exit(1)

    # Sentiment distribution
    sentiment_counts_train = df_train['sentiment'].value_counts()
    print("\nSentiment distribution in Training Data:")
    print(sentiment_counts_train)

    sentiment_counts_test = df_test['sentiment'].value_counts()
    print("\nSentiment distribution in Testing Data:")
    print(sentiment_counts_test)

    # Prepare data
    train_sentences = df_train['sentence'].astype(str).values
    train_labels = df_train['sentiment'].values
    test_sentences = df_test['sentence'].astype(str).values
    test_labels = df_test['sentiment'].values

    # Initialize TfidfVectorizer with dimensionality reduction
    tf_vec = TfidfVectorizer(
        max_features=20000,  # Increased to capture more features
        ngram_range=(1, 2),   # Use unigrams and bigrams
        stop_words='english',
        lowercase=True,
        strip_accents='unicode',
        min_df=50,
        max_df=0.7
    )

    # Transform training data
    print("\nStarting TF-IDF vectorization on training data...")
    start = time()
    X_train_tf = tf_vec.fit_transform(train_sentences)
    end = time()
    print(f"Time to transform training data: {end - start:.2f}s")
    print(f"Training TF-IDF shape: {X_train_tf.shape}")

    # Transform testing data
    print("\nStarting TF-IDF vectorization on testing data...")
    start = time()
    X_test_tf = tf_vec.transform(test_sentences)
    duration = time() - start
    print(f"Time taken to transform testing data: {duration:.2f} seconds")
    print(f"Testing TF-IDF shape: {X_test_tf.shape}")

    # Dimensionality Reduction with TruncatedSVD
    print("\nStarting dimensionality reduction with TruncatedSVD...")
    svd = TruncatedSVD(n_components=feature_dim, random_state=42)
    start = time()
    X_train_svd = svd.fit_transform(X_train_tf)
    X_test_svd = svd.transform(X_test_tf)
    end = time()
    print(f"Time for TruncatedSVD: {end - start:.2f}s")
    print(f"Reduced feature shape: {X_train_svd.shape}")

    # Save the vectorizer and SVD transformer
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as vec_file:
        pickle.dump(tf_vec, vec_file)
    with open(os.path.join(output_dir, 'svd_transformer.pkl'), 'wb') as svd_file:
        pickle.dump(svd, svd_file)
    print("TfidfVectorizer and TruncatedSVD saved.")

    # Convert data to tensors
    print("\nConverting data to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train_svd, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_svd, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("DataLoaders created.")

    # Initialize the model
    input_dim = feature_dim
    hidden_dims = [1024, 512, 256]  # Increased complexity for better learning
    dropout = 0.3  # Reduced dropout to allow more learning
    model = SentimentANN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.to(device)
    print("\nModel architecture:")
    print(model)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Training loop
    print("\nStarting training...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            
            # Print loss every 'print_every' steps
            if (step + 1) % print_every == 0:
                current_loss = loss.item()
                print(f"Epoch [{epoch}/{epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {current_loss:.4f}")

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Step the scheduler
        scheduler.step(avg_loss)

    # Evaluation on test set
    print("\nEvaluating on test set...")
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")
    print(f"AUC-ROC on test set: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(output_dir, 'confusion_matrix_heatmap.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix heatmap saved as '{cm_path}'")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved as '{roc_path}'")

    # Save the model
    model_path = os.path.join(output_dir, 'sentiment_ann_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")

    print("\nTraining and evaluation completed successfully.")

if __name__ == "__main__":
    main(
        dataset_dir='/home/dgxuser16/NTL/mccarthy/ahmad/Projects/ML_Course_Proj/data/twitter', 
        output_dir='output_new', 
        batch_size=1024,       # Increased batch size for efficiency
        epochs=500,              # Reduced epochs; can be adjusted based on validation
        learning_rate=0.001,   # Standard learning rate; consider tuning
        feature_dim=600,
        print_every=5000       # Adjusted to print every 5000 steps to reduce verbosity
    )
