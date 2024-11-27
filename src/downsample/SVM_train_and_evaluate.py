import prepare_data
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from joblib import dump
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

def find_best_threshold(model, X_eval, y_eval, thresholds=np.arange(0.0, 1.0, 0.05)):
    """
    Perform grid search to find the best decision threshold for a given model.
    """
    y_scores = model.decision_function(X_eval)  # Get raw scores for SVM
    metrics_per_threshold = []

    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        precision = precision_score(y_eval, y_pred, zero_division=0)
        recall = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        accuracy = accuracy_score(y_eval, y_pred)
        auc = roc_auc_score(y_eval, y_scores)

        metrics_per_threshold.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "AUC-ROC": auc
        })
    
    # Find the threshold with the highest F1 score
    best_threshold = max(metrics_per_threshold, key=lambda x: x["F1 Score"])
    
    return best_threshold, metrics_per_threshold

def evaluate_model_with_threshold(model, X_eval, y_eval, threshold=0.1):
    """Evaluate model with an adjustable decision threshold."""
    y_scores = model.decision_function(X_eval)
    y_eval_pred = (y_scores > threshold).astype(int)  # Apply threshold

    accuracy = accuracy_score(y_eval, y_eval_pred)
    precision = precision_score(y_eval, y_eval_pred, zero_division=0)
    recall = recall_score(y_eval, y_eval_pred, zero_division=0)
    f1 = f1_score(y_eval, y_eval_pred, zero_division=0)
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
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
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
 
        # Train baseline model
    
    baseline_model = SVC(kernel='linear', C=1.0, class_weight='balanced')
    print("\nTraining baseline model with linear kernel (C=1.0)")
    baseline_model.fit(X_train, y_train, sample_weight=sample_weights_train)
 
    # Evaluate baseline model
    print("\nTraining Data Results for the baseline model")
    evaluate_model_with_threshold(baseline_model, X_train, y_train, threshold=0.3)
    
    
    print("\nEvaluation Results for the baseline model")
    evaluate_model_with_threshold(baseline_model, X_eval, y_eval, threshold=0.3)
   
 
    # Step 1: Search for the best model
    print("\nPerforming grid search to find the best model...")
    param_grid = [
        {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
        {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
    ]
    grid_search = GridSearchCV(
        SVC(class_weight='balanced'),
        param_grid,
        scoring='f1',
        cv=3,  # 3-fold cross-validation for grid search
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train, sample_weight=sample_weights_train)
        
    # Extract the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
 
    # Step 2: Perform cross-validation on the best model
    print("\nPerforming 5-fold cross-validation on the best model...")
    skf = StratifiedKFold(n_splits=5)
    avg_f1, avg_precision, avg_recall = cross_validate_model(best_model, X_train, y_train, cv=skf, threshold=0.2)
    print(f"\nCross-validation results: F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}")
 
    # Step 3: Evaluate the best model on the evaluation set
    print("\nEvaluating the best model on the evaluation set...")
    evaluate_model_with_threshold(best_model, X_eval, y_eval, threshold=0.1)
    
    y_eval_prob = best_model.decision_function(X_eval)
    
    fpr, tpr, thresholds = roc_curve(y_eval, y_eval_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Save the plot
    output_dir = "data/visualization"
    #save the plot
    plt.savefig(os.path.join(output_dir, "roc_curve_SVM_DOWN.png"))
 
    # Save the best model
    dump(best_model, 'models/baseline_model/best_svm_model.joblib')
    print("\nBest model saved as 'best_svm_model.joblib'")
if __name__ == "__main__":
    main()