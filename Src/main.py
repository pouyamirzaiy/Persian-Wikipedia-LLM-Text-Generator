from data_preprocessing import load_data, clean_data, split_data
from feature_engineering import create_preprocessor
from model_training import train_model, evaluate_model
from hyperparameter_tuning import tune_hyperparameters

def main():
    # Load and clean data
    df = load_data('path/to/ad_10000records.csv')
    df = clean_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, 'Clicked on Ad')

    # Feature engineering
    numerical_columns = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']
    categorical_columns = ['Gender', 'Country', 'City', 'Ad Topic Line']
    time_columns = ['Hour', 'Day', 'Month']
    preprocessor = create_preprocessor(numerical_columns, categorical_columns, time_columns)

    # Train and evaluate model
    model = train_model(X_train, y_train, preprocessor)
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    print("Random Forest Classifier:")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Hyperparameter tuning
    grid_search_xgb = tune_hyperparameters(X_train, y_train, preprocessor)
    y_pred_xgb = grid_search_xgb.predict(X_test)
    print("Best Parameters (XGBoost):", grid_search_xgb.best_params_)
    print("Accuracy (XGBoost):", accuracy_score(y_test, y_pred_xgb))
    print("Confusion Matrix (XGBoost):\n", confusion_matrix(y_test, y_pred_xgb))
    print("Classification Report (XGBoost):\n", classification_report(y_test, y_pred_xgb))

if __name__ == "__main__":
    main()
