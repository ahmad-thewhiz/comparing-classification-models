# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from time import time
import argparse
import os
import pickle
import seaborn as sns

def main(dataset_dir, output_dir):
    # Importing the datasets
    try:
        df_train = pd.read_csv(f'{dataset_dir}/train_data.csv')
        df_test = pd.read_csv(f'{dataset_dir}/test_data.csv')
    except FileNotFoundError as e:
        print(f'Train and Test datasets not found in the specified directory: {e}')
        return  

    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")

    print("Training Dataset has {} rows and {} columns".format(df_train.shape[0], df_train.shape[1]))
    print("Testing Dataset has {} rows and {} columns".format(df_test.shape[0], df_test.shape[1]))

    sentiment_counts_train = df_train['sentiment'].value_counts()
    print("\nSentiment distribution in Training Data:")
    print(sentiment_counts_train)

    sentiment_counts_test = df_test['sentiment'].value_counts()
    print("\nSentiment distribution in Testing Data:")
    print(sentiment_counts_test)

    trainX = df_train['sentence']
    trainY = df_train['sentiment']
    testX = df_test['sentence']
    testY = df_test['sentiment']

    # Initialize TfidfVectorizer
    tf_vec = TfidfVectorizer()

    # Transform training data
    start = time()
    X_train_tf = tf_vec.fit_transform(trainX)
    end = time()
    print('\nTime to transform training data: {:.2f}s'.format(end - start))
    print("Training Data Shape: n_samples={}, n_features={}".format(X_train_tf.shape[0], X_train_tf.shape[1]))

    # Transform testing data
    start = time()
    X_test_tf = tf_vec.transform(testX)
    duration = time() - start
    print("\nTime taken to extract features from test data: {:.2f} seconds".format(duration))
    print("Testing Data Shape: n_samples={}, n_features={}".format(X_test_tf.shape[0], X_test_tf.shape[1]))

    # Initialize the Decision Tree Classifier with provided parameters
    dt_classifier = DecisionTreeClassifier(random_state=42, criterion='gini', max_features=None)

    # Train the Decision Tree Classifier
    print("\nStarting Training for Decision Tree...")
    start = time()
    dt_classifier.fit(X_train_tf, trainY)
    end = time()
    print(f"Training completed in {end - start:.2f}s")

    # Evaluate the model
    start = time()
    y_pred = dt_classifier.predict(X_test_tf)
    y_pred_proba = dt_classifier.predict_proba(X_test_tf)[:, 1]
    prediction_time = time() - start
    print("\nPrediction time: {:.4f}s".format(prediction_time))

    # Calculate Accuracy
    acc = metrics.accuracy_score(testY, y_pred)
    print(f"Accuracy on test set: {acc*100:.2f}%")

    # Calculate AUC-ROC
    auc = roc_auc_score(testY, y_pred_proba)
    print(f"AUC-ROC on test set: {auc:.4f}")

    print("\nClassification report for the classifier: \n")
    print(metrics.classification_report(testY, y_pred))

    # Create a confusion matrix heatmap
    conf_matrix = confusion_matrix(testY, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f'{output_dir}/confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix heatmap saved as 'confusion_matrix_heatmap.png'")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(testY, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("ROC curve saved as 'roc_curve.png'")

    # Save the model and the vectorizer
    with open(f'{output_dir}/dt_classifier_model.pkl', 'wb') as model_file:
        pickle.dump(dt_classifier, model_file)
    with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(tf_vec, vec_file)

    print("Model saved as 'dt_classifier_model.pkl'")
    print("TfidfVectorizer saved as 'tfidf_vectorizer.pkl'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and save output.")
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help="Path to the dataset directory.")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to the output directory.")

    args = parser.parse_args()

    main(args.dataset_dir, args.output_dir)
