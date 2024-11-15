from sklearn.ensemble import RandomForestClassifier

def create_random_forest_model(n_estimators=100, max_depth=None, class_weight="balanced", random_state=42):
    """
    Creates and returns a Random Forest model with specified configuration, including the number of estimators,
    max depth, class weight handling for imbalanced classes, and random state for reproducibility.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                   class_weight=class_weight, random_state=random_state)
    print(f"Random Forest model created with n_estimators={n_estimators}, max_depth={max_depth}, class_weight={class_weight}, random_state={random_state}")
    return model

if __name__ == "__main__":
    model = create_random_forest_model()
    print("Random Forest model created with number of estimators:", model.n_estimators)