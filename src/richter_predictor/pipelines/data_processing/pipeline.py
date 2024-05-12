from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_raw_data, define_categorical_columns

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_raw_data,
                inputs=["train_values", "test_values"],
                outputs="df",
                name="load_raw_data",
            ),
            node(
                func=define_categorical_columns,
                inputs=["df", "params:non_categorical_columns"],
                outputs="categorical_columns",
                name="define_categorical_columns",
            )
        ]
    )
