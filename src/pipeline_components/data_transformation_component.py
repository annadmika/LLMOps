"""Data transformation component for Vertex AI pipeline."""

from kfp.dsl import OutputPath, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.3.2",
        "datasets==4.0.0",
        "gcsfs",
    ],
)
def data_transformation_component(
    raw_dataset_uri: str,
    train_test_split_ratio: float,
    train_dataset: OutputPath("Dataset"),  # type: ignore
    test_dataset: OutputPath("Dataset"),  # type: ignore
) -> None:
    """Format and split Tarot Reading dataset for Phi-3 fine-tuning."""
    import logging

    import pandas as pd
    from datasets import Dataset

    def format_dataset_to_phi_messages(dataset: Dataset) -> Dataset:
        """Format dataset to Phi messages structure."""

        def format_dataset(examples):
            """Format a single example to Phi messages structure."""
            converted_sample = [
                {"role": "user", "content": examples["prompt"]},
                {"role": "assistant", "content": examples["response"]},
            ]
            return {"messages": converted_sample}

        return (
            dataset.map(format_dataset)
            .remove_columns(["prompt", "response"])
        )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting data transformation process...")

    logger.info(f"Reading from {raw_dataset_uri}")
    dataset = Dataset.from_pandas(pd.read_csv(raw_dataset_uri))

    logger.info("Formatting and splitting dataset...")
    formatted_dataset = format_dataset_to_phi_messages(dataset)
    split_dataset = formatted_dataset.train_test_split(test_size=train_test_split_ratio)

    # OutputPath provides a directory, so we need to append a filename
    import os
    train_file = os.path.join(train_dataset, "train_dataset.csv")
    test_file = os.path.join(test_dataset, "test_dataset.csv")
    
    logger.info(f"Writing train dataset to {train_file}...")
    split_dataset["train"].to_csv(train_file, index=False)

    logger.info(f"Writing test dataset to {test_file}...")
    split_dataset["test"].to_csv(test_file, index=False)

    logger.info("Data transformation process completed successfully")
