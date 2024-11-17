# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import argparse
import os
import sys

def main(dataset_dir, output_dir):
    # Importing the datasets
    try:
        df_train = pd.read_csv(os.path.join(dataset_dir, 'train_data.csv')).sample(n=500000, random_state=42)
        df_test = pd.read_csv(os.path.join(dataset_dir, 'test_data.csv'))
    except FileNotFoundError as e:
        print(f'Error: {e}')
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        print(f'Error: One of the CSV files is empty or malformed: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'An unexpected error occurred while reading the datasets: {e}')
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")

    print(f"\nTraining data shape: {df_train.shape}\n")
    print(f"Testing data shape: {df_test.shape}")

    if 'sentiment' not in df_train.columns or 'sentiment' not in df_test.columns:
        print("Error: 'sentiment' column not found in one of the datasets.")
        sys.exit(1)

    trainX = df_train['sentence'].astype(str)
    trainY = df_train['sentiment']
    testX = df_test['sentence'].astype(str)
    testY = df_test['sentiment']

    tf_vec = TfidfVectorizer()
    print("\nStarting TF-IDF vectorization on training data...")
    start = time()
    X_train_tf = tf_vec.fit_transform(trainX)
    end = time()
    print(f"Time to transform training data: {end - start:.2f}s")

    print("\nStarting TF-IDF vectorization on testing data...")
    start = time()
    X_test_tf = tf_vec.transform(testX)
    print(f"Time taken to extract features from test data: {time() - start:.2f}s")

    # Initialize Random Forest Classifier with default parameters
    rf_classifier = RandomForestClassifier(random_state=42, bootstrap=True, criterion='entropy', max_features='log2', n_estimators=300)
    
    print("\nTraining Random Forest Classifier...")
    start = time()
    try:
        rf_classifier.fit(X_train_tf, trainY)
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        sys.exit(1)
    print(f"Training completed in {time() - start:.2f}s")

    print("\nMaking predictions on the test set...")
    start = time()
    try:
        y_pred = rf_classifier.predict(X_test_tf)
        if hasattr(rf_classifier, "predict_proba"):
            y_pred_proba = rf_classifier.predict_proba(X_test_tf)[:, 1]
        else:
            y_pred_proba = rf_classifier.decision_function(X_test_tf)
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        sys.exit(1)
    print(f"Prediction time: {time() - start:.2f}s")

    acc = metrics.accuracy_score(testY, y_pred)
    print(f"\nAccuracy on test set: {acc*100:.2f}%")

    try:
        auc = roc_auc_score(testY, y_pred_proba)
        print(f"AUC-ROC on test set: {auc:.4f}")
    except ValueError as e:
        print(f"Error computing AUC-ROC: {e}")
        auc = None

    print("\nClassification report for the Random Forest classifier: \n")
    print(metrics.classification_report(testY, y_pred))

    conf_matrix = confusion_matrix(testY, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    cm_path = os.path.join(output_dir, 'confusion_matrix_heatmap.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix heatmap saved as '{cm_path}'")

    if auc is not None:
        fpr, tpr, thresholds = roc_curve(testY, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved as '{roc_path}'")
    else:
        print("ROC curve was not plotted due to inability to compute AUC-ROC.")

    try:
        model_path = os.path.join(output_dir, 'rf_classifier_model.pkl')
        with open(model_path, 'wb') as model_file:
            pickle.dump(rf_classifier, model_file)
        print(f"Model saved as '{model_path}'")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

    try:
        vec_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
        with open(vec_path, 'wb') as vec_file:
            pickle.dump(tf_vec, vec_file)
        print(f"TfidfVectorizer saved as '{vec_path}'")
    except Exception as e:
        print(f"An error occurred while saving the vectorizer: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and save output.")
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help="Path to the dataset directory.")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to the output directory.")

    args = parser.parse_args()

    main(args.dataset_dir, args.output_dir)
