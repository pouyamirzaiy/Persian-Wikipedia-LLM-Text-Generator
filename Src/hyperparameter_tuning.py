from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def tune_hyperparameters(X_train, y_train, preprocessor):
    xgb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])

    param_grid_xgb = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0]
    }

    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, n_jobs=1, scoring='accuracy', verbose=2)
    grid_search_xgb.fit(X_train, y_train)
    return grid_search_xgb
