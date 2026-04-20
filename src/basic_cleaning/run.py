#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    # Start a W&B run
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    logger.info("Downloading input artifact")

    # Download input artifact (and register lineage)
    artifact = run.use_artifact(args.input_artifact)
    artifact_local_path = artifact.file()

    # Load dataset
    df = pd.read_csv(artifact_local_path)

    logger.info("Applying basic data cleaning")

    # Remove price outliers
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=["price"])
    df = df[df["price"].between(args.min_price, args.max_price)]

    # Boundary filter
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Cleaned data has %s rows and %s columns", *df.shape)

    # Save cleaned data
    cleaned_filename = "clean_sample.csv"
    df.to_csv(cleaned_filename, index=False)

    logger.info("Uploading cleaned dataset as artifact")

    # Create and log output artifact
    output_artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    output_artifact.add_file(cleaned_filename)
    run.log_artifact(output_artifact)

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input dataset artifact to clean",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output cleaned dataset artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact (e.g. cleaned_data)",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed price (EDA-based)",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed price (EDA-based)",
        required=True,
    )

    args = parser.parse_args()

    go(args)
