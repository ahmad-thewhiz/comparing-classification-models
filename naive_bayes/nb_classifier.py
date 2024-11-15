# Importing the libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import pickle

import seaborn as sns
from sklearn.metrics import confusion_matrix

from time import time

import argparse
import os


def main(dataset_dir, output_dir):

    # Importing the datasets
    try:
        df_train = pd.read_csv(f'{dataset_dir}/train_data.csv')
        df_test = pd.read_csv(f'{dataset_dir}/test_data.csv')
    except FileNotFoundError as e:
        print(f'Train and Test datasets not found in the specified directory: {e}')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")

    print("Training Dataset has {} rows and {} columns".format(df_train.shape[0], df_train.shape[1]))
    print("Testing Dataset has {} rows and {} columns".format(df_test.shape[0], df_test.shape[1]))

    df_train.head(5)
    sentiment_counts_train = df_train['sentiment'].value_counts()
    print(sentiment_counts_train)

    df_test.head(5)
    sentiment_counts_test = df_test['sentiment'].value_counts()
    print(sentiment_counts_test)

    trainX = df_train['sentence']
    trainY = df_train['sentiment']
    testX = df_test['sentence']
    testY = df_test['sentiment']

    tf_vec = TfidfVectorizer()
    start = time()
    X_train_tf = tf_vec.fit_transform(trainX)
    end = time()
    print('Time to transform training data: {}s'.format(end - start))
    print(f"n_samples: %d, n_features: %d", X_train_tf.shape)

    start = time()
    X_test_tf = tf_vec.transform(testX)
    duration = time() - start
    print("Time taken to extract features from test data : %f seconds" % (duration))
    print("n_samples: %d, n_features: %d" % X_test_tf.shape)


    # Defining the parameter grid for Bernoulli Naive Bayes

    param_grid = {
        'alpha': [0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1, 3, 5, 8, 10],  # Smoothing parameter
        'binarize': [None, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0],  # Threshold for binarization
        'fit_prior': [True, False]  # Whether to learn class prior probabilities
    }


    # Initialize the BernoulliNB model
    nb_classifier = BernoulliNB()


    # Setting up GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=nb_classifier, 
                            param_grid=param_grid, 
                            scoring='accuracy', 
                            cv=5, 
                            verbose=1, 
                            n_jobs=-1) 


    # Start the grid search
    start = time()
    grid_search.fit(X_train_tf, trainY)
    end = time()
    print(f"GridSearchCV completed in {end - start:.2f}s")

    print("Best parameters found:")
    print(grid_search.best_params_)

    print("Best cross-validation accuracy:")
    print(grid_search.best_score_)


    # Evaluate the model with the best parameters on the test set
    best_nb_classifier = grid_search.best_estimator_
    start = time()
    y1_predict = best_nb_classifier.predict(X_test_tf)
    prediction_time = time() - start
    print("Prediction time: %fs" % prediction_time)

    acc = metrics.accuracy_score(testY, y1_predict)
    print(f"Accuracy on test set: {(acc*100):.2f}%")

    print("Classification report for the optimized classifier: \n")
    print(metrics.classification_report(testY, y1_predict))


    # Create a heatmap
    conf_matrix = confusion_matrix(testY, y1_predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig(f'{output_dir}/confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')


    # Save the model
    with open(f'{output_dir}/best_nb_classifier_model.pkl', 'wb') as model_file:
        pickle.dump(best_nb_classifier, model_file)

    print("Model saved as 'best_nb_classifier_model.pkl'")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process dataset and save output.")

    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help="Path to the dataset directory.")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to the output directory.")

    args = parser.parse_args()

    main(args.dataset_dir, args.output_dir)
