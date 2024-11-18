import os
import pandas as pd
import numpy as np
from collections import Counter
import shutil
import argparse

def duplicate_feature_file(original_path, downsampled_dir):
    """
    Duplicates a feature file without modifying the filename and saves it in the specified directory.

    Args:
        original_path (str): Path to the original feature file.
        downsampled_dir (str): Directory to save the duplicated file.

    Returns:
        str: Path to the duplicated file.
    """
    downsampled_path = os.path.join(downsampled_dir, os.path.basename(original_path))
    shutil.copy(original_path, downsampled_path)
    return downsampled_path


def downsample_to_match_event_data(df, contextual_feature_dir, label_dir, downsampled_feature_dir, downsampled_label_dir):
    """
    Resamples the dataset by keeping all event data and adding non-event data to match the event count.

    Args:
        df (pd.DataFrame): DataFrame with label and feature metadata.
        contextual_feature_dir (str): Directory for original contextual feature files.
        label_dir (str): Directory for original label files.
        downsampled_feature_dir (str): Directory to save duplicated contextual feature files.
        downsampled_label_dir (str): Directory to save duplicated label files.

    Returns:
        pd.DataFrame: Resampled DataFrame with balanced event and non-event data.
    """
    # Set binary class based on 'label_list'
    df['class'] = df['label_list'].apply(lambda x: 1 if "Phonological" in x else 0)

    # Separate event and non-event samples
    event_samples = df[df['class'] == 1].copy()
    non_event_samples = df[df['class'] == 0].copy()

    print(f"Initial class distribution: {Counter(df['class'])}")

    # Number of event samples
    num_event_samples = len(event_samples)

    # Randomly select non-event samples to match the number of event samples
    selected_non_event_samples = non_event_samples.sample(n=num_event_samples, random_state=0)

    # Combine event and non-event samples
    downsampled_df = pd.concat([event_samples, selected_non_event_samples])

    print(f"New class distribution after resampling: {Counter(downsampled_df['class'])}")

    # Shuffle the combined DataFrame to ensure a random distribution
    downsampled_df = downsampled_df.sample(frac=1, random_state=0).reset_index(drop=True)

    # Generate new files in resampled directories
    downsampled_data = []
    for _, row in downsampled_df.iterrows():
        # Duplicate feature and label files
        downsampled_feature_file = duplicate_feature_file(row['contextual_feature_file'], downsampled_feature_dir)
        downsampled_label_file = duplicate_feature_file(row['contextual_label_file'], downsampled_label_dir)

        # Append the updated paths and data to the new resampled dataset
        row_data = row.to_dict()
        row_data['contextual_feature_file'] = downsampled_feature_file
        row_data['contextual_label_file'] = downsampled_label_file
        downsampled_data.append(row_data)

    # Convert list of dictionaries to DataFrame
    downsampled_df = pd.DataFrame(downsampled_data)

    print("Resampling completed. Total entries in resampled dataset:", len(downsampled_df))

    return downsampled_df


def save_downsampled_metadata(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Downsampled metadata saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downsample dataset to balance event and non-event data.")
    parser.add_argument('--label_info_path', type=str, required=True, help="Path to the label_info_context.csv.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the downsampled label_info.csv.")
    parser.add_argument('--contextual_feature_dir', type=str, required=True, help="Directory containing the contextual feature files.")
    parser.add_argument('--label_dir', type=str, required=True, help="Directory containing the original label files.")
    parser.add_argument('--downsampled_feature_dir', type=str, required=True, help="Directory to save duplicated contextual feature files.")
    parser.add_argument('--downsampled_label_dir', type=str, required=True, help="Directory to save duplicated label files.")

    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.label_info_path)

    # Ensure output directories exist
    os.makedirs(args.downsampled_feature_dir, exist_ok=True)
    os.makedirs(args.downsampled_label_dir, exist_ok=True)

    # Perform downsampling
    downsampled_df = downsample_to_match_event_data(
        df,
        args.contextual_feature_dir,
        args.label_dir,
        args.downsampled_feature_dir,
        args.downsampled_label_dir
    )

    # Save the downsampled metadata
    save_downsampled_metadata(downsampled_df, args.output_path)
