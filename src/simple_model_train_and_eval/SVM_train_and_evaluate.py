import prepare_data
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from joblib import dump
import numpy as np
from sklearn.metrics import roc_curve, auc
import time
from sklearn.ensemble import BaggingClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def evaluate_model_with_threshold(model, X_eval, y_eval, threshold=0.1):
    """Evaluate model with an adjustable decision threshold."""
    y_scores = model.decision_function(X_eval)
    y_eval_pred = (y_scores > threshold).astype(int)  # Apply threshold

    accuracy = accuracy_score(y_eval, y_eval_pred)
    precision = precision_score(y_eval, y_eval_pred, average='binary')
    recall = recall_score(y_eval, y_eval_pred, average='binary')
    f1 = f1_score(y_eval, y_eval_pred, average='binary')
    auc = roc_auc_score(y_eval, y_scores)
    conf_matrix = confusion_matrix(y_eval, y_eval_pred)

    print(classification_report(y_eval, y_eval_pred))
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-ROC: {auc}")
    print("Confusion Matrix:")
    print(conf_matrix)
    return accuracy, precision, recall, f1, auc, conf_matrix

def cross_validate_model(model, X, y, cv, threshold=0.1):
    """Perform cross-validation manually and return the average scores for F1, precision, and recall."""
    f1_scores = []
    precision_scores = []
    recall_scores = []
    fold_results = []  # To store results per fold

    for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit model on training fold
        model.fit(X_train, y_train)
        
        # Get predictions with threshold on test fold
        y_scores = model.decision_function(X_test)
        y_pred = (y_scores > threshold).astype(int)
        
        # Calculate metrics on the test fold
        f1 = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        
        # Append metrics to lists
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        # Print fold results
        print(f"Fold {fold_idx}: F1 Score={f1}, Precision={precision}, Recall={recall}")
        
        # Store each fold's metrics in a dict
        fold_results.append({
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        })
    
    # Print all fold results (optional)
    print("All fold results:", fold_results)
    
    # Calculate and return the average of each metric across folds
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    
    return avg_f1, avg_precision, avg_recall

def main():
 
    # Load data
    target_length = prepare_data.get_max_sequence_length("data/metadata/label_downsampled.csv", "data/downsampled_features")
    X_train, y_train, sample_weights_train, scaler = prepare_data.load_train_data(
        label_info_path="data/metadata/train_downsample.csv",
        primary_feature_dir="data/downsampled_features",
        target_length=target_length
    )
    X_eval, y_eval = prepare_data.load_eval_data(
        label_info_path="data/metadata/eval_downsample.csv",
        primary_feature_dir="data/downsampled_features",
        target_length=target_length,
        scaler=scaler
    )
    n_estimators = 10
    baseline_model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    baseline_model.fit(X_train, y_train)
    print("\nTraining Data Results for the baseline model")
    evaluate_model_with_threshold(baseline_model, X_train, y_train, threshold=0.1)
    
    print("\nEvaluation Results for the baseline model")
    evaluate_model_with_threshold(baseline_model, X_eval, y_eval, threshold=0.1)
 
if __name__ == "__main__":
    main()