import prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_validate, KFold
from joblib import dump
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def evaluate_cross_validation(model, X, y, cv_folds):
    """
    Perform cross-validation and print precision, recall, and F1 scores for each fold.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Perform cross-validation and collect precision, recall, and F1 scores
    cv_results = cross_validate(
        model, X, y, cv=kf,
        scoring=['precision_macro', 'recall_macro', 'f1_macro'],
        return_train_score=False
    )

    # Extract and print scores for each fold
    precision_scores = cv_results['test_precision_macro']
    recall_scores = cv_results['test_recall_macro']
    f1_scores = cv_results['test_f1_macro']

    print("Cross-Validation Scores for Each Fold:")
    for fold, (precision, recall, f1) in enumerate(zip(precision_scores, recall_scores, f1_scores), 1):
        print(f"Fold {fold}: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

    # Print average scores
    print("\nTraining Results --- Average Cross-Validation Scores:")
    print(
        f"Precision: {precision_scores.mean():.4f}, Recall: {recall_scores.mean():.4f}, F1 Score: {f1_scores.mean():.4f}")


def main():
    # Calculate target_length from training data
    target_length = prepare_data.get_max_sequence_length(
        label_info_path="data/metadata/label_downsampled.csv",
        primary_feature_dir="data/downsampled_features"
    )
    print(f"Using target length of {target_length} based on maximum sequence length in training data.")

    # Load training data
    X_train, y_train, sample_weights_train, scaler = prepare_data.load_train_data(
        label_info_path="data/metadata/train_downsample.csv",
        primary_feature_dir="data/downsampled_features",
        target_length=target_length
    )

    # Load evaluation data
    X_eval, y_eval = prepare_data.load_eval_data(
        label_info_path="data/metadata/eval_downsample.csv",
        primary_feature_dir="data/downsampled_features",
        target_length=target_length,
        scaler=scaler
    )

    # Initialize Logistic Regression model
    model = LogisticRegression(class_weight='balanced', max_iter=1000)

    # Perform cross-validation and evaluate
    print("\nRunning Cross-Validation on Logistic Regression Model...")
    evaluate_cross_validation(model, X_train, y_train, cv_folds=5)

    # Train the model on the entire training set
    print("\nTraining Logistic Regression on the training set...")
    model.fit(X_train, y_train, sample_weight=sample_weights_train)

    # Evaluate on the evaluation set using a customizable threshold
    custom_threshold = 0.3  # You can change this value to your desired threshold
    print(
        f"\nEvaluation Results for the Logistic Regression Model with Custom Threshold (Threshold = {custom_threshold}):")

    # Get probabilities for the positive class
    y_eval_prob = model.predict_proba(X_eval)[:, 1]

    # Apply the custom threshold to classify the samples
    y_eval_pred_custom = (y_eval_prob >= custom_threshold).astype(int)

    # Print the classification report
    print(classification_report(y_eval, y_eval_pred_custom))

    # Save the trained model

    print("\nModel saved as best_logistic_model.joblib")

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
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "roc_auc_curve_LR_DOWN.png")
    plt.savefig(output_path)
    plt.close()
    dump(model, 'models/baseline_model/best_logistic_model.joblib')


if __name__ == "__main__":
    main()
