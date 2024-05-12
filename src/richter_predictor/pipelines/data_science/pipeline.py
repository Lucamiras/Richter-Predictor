from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_catboost_classifier, generate_predictions

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_catboost_classifier,
                inputs=[
                    "df", 
                    "train_labels",
                    "params:param_grid",
                    "categorical_columns",
                    "params:eval_metric",
                    "params:predefined_params",
                    "params:finetune",
                    ],
                outputs="results",
                name="build_classifier",
            ),
            node(
                func=generate_predictions,
                inputs=["df", "submission_format", "results"],
                outputs=None,
                name="generate_predictions",
            )
        ]
    )
