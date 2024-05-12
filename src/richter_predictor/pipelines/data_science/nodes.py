import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from catboost import CatBoostClassifier

def build_catboost_classifier(df: pd.DataFrame, train_labels: pd.DataFrame, params: dict, categorical_columns: list, eval_metric: str, predefined_params: dict, finetune: bool) -> dict:
    """
    Fine-tunes a CatBoost model using the provided dataframe and parameters.

    Args:
        df (pd.DataFrame): The dataframe containing the training data.
        params (dict): The parameters to be used for fine-tuning.

    Returns:
        dict: A dictionary containing the results of the fine-tuning process.
    """

    logger = logging.getLogger(__name__)
    
    results = {}

    X = df.query('is_train == True').drop(['is_train'], axis=1)
    y = train_labels['damage_grade']

    if finetune:
        logger.info(msg="Finetuning is enabled. Starting fine-tuning process...")

        model = CatBoostClassifier(eval_metric=eval_metric,
                                   cat_features=categorical_columns,
                                   task_type="GPU")
        
        grid_search_cv = GridSearchCV(model,
                                      params,
                                      cv=3,
                                      verbose=3)

        grid_search_cv.fit(X, y)

        results['best_estimator'] = grid_search_cv.best_estimator_
        results['best_score'] = grid_search_cv.best_score_
        results['best_params'] = grid_search_cv.best_params_
        
        logger.info(f"Found best parameters: {results['best_params']}")
        
    else:
        logger.info(msg="Finetuning is disabled. Skipping and using predefined parameters...")
        
        model = CatBoostClassifier(eval_metric=eval_metric,
                                   cat_features=categorical_columns,
                                   task_type="GPU",
                                   depth=predefined_params['depth'],
                                   iterations=predefined_params['iterations'],
                                   learning_rate=predefined_params['learning_rate'],
                                   l2_leaf_reg=predefined_params['l2_leaf_reg'],
                                   border_count=predefined_params['border_count'])
        
        # cross_val_scores = np.mean(cross_val_score(model, X, y, cv=5, scoring='f1_micro'))
        
        # logger.info(f"Trained classifier with cross_val_score of {cross_val_scores}.")

        results['best_estimator'] = model.fit(X, y)
        # results['best_score'] = cross_val_scores
    
    return results

def generate_predictions(df: pd.DataFrame, submission_format: pd.DataFrame, results: dict) -> None:
    
    test_features = df.query('is_train == False').drop('is_train', axis=1)
    
    model = results['best_estimator']
    
    predictions = model.predict(test_features)
    
    submission_format['damage_grade'] = predictions

    submission_format.to_csv('data/07_model_output/submission.csv', index=False)