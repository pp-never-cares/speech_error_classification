import prepare_data
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from joblib import dump
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

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
    target_length = prepare_data.get_max_sequence_length("data/metadata/label_train_resampled.csv", "data/contextual_features")
    X_train, y_train, sample_weights_train, scaler = prepare_data.load_train_data(
        target_length=target_length,
        primary_feature_dir="data/contextual_features",
        secondary_feature_dir="data/resampled_features"
    )
    X_eval, y_eval = prepare_data.load_eval_data(
        target_length=target_length,
        primary_feature_dir="data/contextual_features",
        scaler=scaler
    )

    # Baseline model with linear kernel
    baseline_model = SVC(kernel='linear', C=1.0, class_weight='balanced')
    print("\nTraining baseline model with linear kernel (C=1.0)")
    baseline_model.fit(X_train, y_train, sample_weight=sample_weights_train)
    
    print("\nEvaluation Results for the baseline model")
    evaluate_model_with_threshold(baseline_model, X_eval, y_eval, threshold=0.1)
    best_prameters = {'C': 1.0, 'gamma': 0.01, 'kernel': 'rbf'}
    best_model = SVC(**best_prameters)
    print("\nTraining best model with RBF kernel")
    best_model.fit(X_train, y_train, sample_weight=sample_weights_train)
    # Evaluate the best model
    print("\nEvaluation Results for the best model")
    evaluate_model_with_threshold(best_model, X_eval, y_eval, threshold=0.1)
    
    #print auc-roc curve
    y_scores = best_model.decision_function(X_eval)
    fpr, tpr, thresholds = roc_curve(y_eval, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    output_dir = "data/visualization"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "roc_auc_curve_SVM_simple.png")
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    main()
