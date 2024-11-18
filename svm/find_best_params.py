# Importing the libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC  # Changed from BernoulliNB to SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
# Removed GridSearchCV as it's no longer needed

import pickle

import seaborn as sns
from sklearn.metrics import confusion_matrix

from time import time

import argparse
import os


def main(dataset_dir, output_dir):

    # Importing the datasets
    try:
        df_train = pd.read_csv(f'{dataset_dir}/train_data.csv').sample(n=50000, random_state=42)
        df_test = pd.read_csv(f'{dataset_dir}/test_data.csv')
    except FileNotFoundError as e:
        print(f'Train and Test datasets not found in the specified directory: {e}')
        return  # Exit the function if datasets are not found

    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")

    print("Training Dataset has {} rows and {} columns".format(df_train.shape[0], df_train.shape[1]))
    print("Testing Dataset has {} rows and {} columns".format(df_test.shape[0], df_test.shape[1]))

    print("\nFirst 5 rows of Training Dataset:")
    print(df_train.head(5))
    sentiment_counts_train = df_train['sentiment'].value_counts()
    print("\nSentiment distribution in Training Dataset:")
    print(sentiment_counts_train)

    print("\nFirst 5 rows of Testing Dataset:")
    print(df_test.head(5))
    sentiment_counts_test = df_test['sentiment'].value_counts()
    print("\nSentiment distribution in Testing Dataset:")
    print(sentiment_counts_test)

    trainX = df_train['sentence']
    trainY = df_train['sentiment']
    testX = df_test['sentence']
    testY = df_test['sentiment']

    # tf_vec = TfidfVectorizer()
    tf_vec = TfidfVectorizer(
    min_df=10,           # Ignore terms that appear in fewer than 5 documents
    max_df=0.7          # Ignore terms that appear in more than 70% of documents
    )

    start = time()
    X_train_tf = tf_vec.fit_transform(trainX)
    end = time()
    print('\nTime to transform training data: {:.2f}s'.format(end - start))
    print("n_samples: {}, n_features: {}".format(X_train_tf.shape[0], X_train_tf.shape[1]))

    start = time()
    X_test_tf = tf_vec.transform(testX)
    duration = time() - start
    print("Time taken to extract features from test data: {:.2f} seconds".format(duration))
    print("n_samples: {}, n_features: {}".format(X_test_tf.shape[0], X_test_tf.shape[1]))

    # Initialize the SVM classifier with predefined hyperparameters
    # You can adjust 'C', 'kernel', and 'gamma' as needed
    svm_classifier = SVC(
        C=1.0,            # Regularization parameter
        kernel='rbf',  # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid', etc.
        gamma=1.0,    # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        random_state=42    # Seed for reproducibility
    )

    # Train the SVM classifier
    print("\nTraining the SVM classifier...")
    start = time()
    svm_classifier.fit(X_train_tf, trainY)
    end = time()
    print(f"SVM training completed in {end - start:.2f}s")

    # Evaluate the model on the test set
    start = time()
    y_pred = svm_classifier.predict(X_test_tf)
    prediction_time = time() - start
    print("Prediction time: {:.4f}s".format(prediction_time))

    acc = metrics.accuracy_score(testY, y_pred)
    print(f"Accuracy on test set: {acc*100:.2f}%")

    print("\nClassification report for the SVM classifier: \n")
    print(metrics.classification_report(testY, y_pred))

    # Create a heatmap for the confusion matrix
    conf_matrix = confusion_matrix(testY, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])

    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig(f'{output_dir}/confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    # Save the model and the vectorizer
    with open(f'{output_dir}/svm_classifier_model.pkl', 'wb') as model_file:
        pickle.dump(svm_classifier, model_file)
    
    with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(tf_vec, vec_file)

    print("Model saved as 'svm_classifier_model.pkl'")
    print("TF-IDF Vectorizer saved as 'tfidf_vectorizer.pkl'")

main('/home/dgxuser16/NTL/mccarthy/ahmad/Projects/ML_Course_Proj/data/twitter', 'output')