import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from time import time

def main(dataset_dir, output_dir):

    # Importing the datasets
    try:
        df_train = pd.read_csv(os.path.join(dataset_dir, 'train_data.csv'))
        df_test = pd.read_csv(os.path.join(dataset_dir, 'test_data.csv'))
    except FileNotFoundError as e:
        print(f'Train and Test datasets not found in the specified directory: {e}')
        return  # Exit the function if datasets are not found

    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")

    print(f"Training Dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns")
    print(f"Testing Dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns")

    print("First 5 rows of the Training Dataset:")
    print(df_train.head(5))
    sentiment_counts_train = df_train['sentiment'].value_counts()
    print("Sentiment distribution in Training Set:")
    print(sentiment_counts_train)

    print("First 5 rows of the Testing Dataset:")
    print(df_test.head(5))
    sentiment_counts_test = df_test['sentiment'].value_counts()
    print("Sentiment distribution in Testing Set:")
    print(sentiment_counts_test)

    trainX = df_train['sentence']
    trainY = df_train['sentiment']
    testX = df_test['sentence']
    testY = df_test['sentiment']

    # Vectorization
    tf_vec = TfidfVectorizer()
    start = time()
    X_train_tf = tf_vec.fit_transform(trainX)
    end = time()
    print(f'Time to transform training data: {end - start:.2f}s')
    print(f"Training data shape: {X_train_tf.shape}")

    start = time()
    X_test_tf = tf_vec.transform(testX)
    duration = time() - start
    print(f"Time taken to extract features from test data: {duration:.2f} seconds")
    print(f"Testing data shape: {X_test_tf.shape}")

    # Defining the parameter grid for Logistic Regression
    param_distributions = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga', 'lbfgs', 'sag'],
    'tol': [1e-4, 1e-3],
    'max_iter': [200, 300],
    'l1_ratio': [0.1, 0.5, 0.9]  # Only relevant for 'elasticnet'
    }

    # Initialize the LogisticRegression model
    LRmodel = LogisticRegression()

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=LRmodel,
        param_distributions=param_distributions,
        n_iter=20, 
        scoring='f1_weighted',
        cv=3,  
        verbose=2,
        n_jobs=20,
        random_state=42,
        return_train_score=False
    )
    

    # Start the grid search
    print("Starting RandomizedSearch...")
    start = time()
    random_search.fit(X_train_tf, trainY)
    end = time()
    print(f"RandomizedSearch completed in {end - start:.2f}s")

    print("Best parameters found:")
    print(random_search.best_params_)

    print("Best cross-validation F1-weighted score:")
    print(f"{random_search.best_score_:.4f}")

    # Evaluate the model with the best parameters on the test set
    best_lr_classifier = random_search.best_estimator_
    start = time()
    y1_predict = best_lr_classifier.predict(X_test_tf)
    prediction_time = time() - start
    print(f"Prediction time: {prediction_time:.4f}s")

    acc = metrics.accuracy_score(testY, y1_predict)
    print(f"Accuracy on test set: {acc * 100:.2f}%")

    print("Classification report for the optimized classifier:\n")
    print(metrics.classification_report(testY, y1_predict))

    # Create a heatmap
    conf_matrix = confusion_matrix(testY, y1_predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Negative', 'Positive'], 
        yticklabels=['Negative', 'Positive']
    )

    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    heatmap_path = os.path.join(output_dir, 'confusion_matrix_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix heatmap saved as '{heatmap_path}'")

    # Save the model
    model_path = os.path.join(output_dir, 'best_lr_classifier_model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(best_lr_classifier, model_file)

    print(f"Model saved as '{model_path}'")

if __name__ == "__main__":
    # Example usage:
    # main('/path/to/dataset', '/path/to/output')

    # For demonstration purposes, replace the paths below with actual paths
    dataset_directory = '/home/dgxuser16/NTL/mccarthy/ahmad/Projects/ML_Course_Proj/data/twitter'
    output_directory = 'output/'

    main(dataset_directory, output_directory)
