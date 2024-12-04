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
    print("This SVM model is trained with a Bagging Classifier, similar to the downsampled model. Please refer to the downsampled model for more details.")
    baseline_model.fit(X_train, y_train)
    print("\nTraining Data Results for the baseline model")
    evaluate_model_with_threshold(baseline_model, X_train, y_train, threshold=0.1)
    
    print("\nEvaluation Results for the baseline model")
    evaluate_model_with_threshold(baseline_model, X_eval, y_eval, threshold=0.1)
 
if __name__ == "__main__":
    main()