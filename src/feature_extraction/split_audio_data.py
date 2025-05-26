"""
Split data into training, evaluation, and testing sets based on unique sounds.

This script:
1. Reads all unique sound identifiers from the label_info.csv.
2. Splits the set of unique sounds into train, eval, and test (based on the provided ratios).
3. Reads each row from label_info.csv:
    - If the label is in the list of events to consider, keep it as is.
    - Otherwise, treat it as a non-event (e.g., label as "NonEvent").
    - Assign the row to train, eval, or test depending on which split its sound belongs to.
4. Saves the rows to train.csv, eval.csv, and test.csv.

Usage:
    python split_data.py --label_info_path <label_info_path> --output_dir <output_dir> --eval_ratio <eval_ratio> --test_ratio <test_ratio>

    - label_info_path (str): The path to the label_info.csv file.
    - output_dir (str): The directory to save the split CSV files.
    - eval_ratio (float): The ratio of sounds to include in the evaluation set.
    - test_ratio (float): The ratio of sounds to include in the testing set.
    
Example:
    python split_data.py --label_info_path /data/metadata/label_info.csv --output_dir /data/metadata --eval_ratio 0.1 --test_ratio 0.1
"""

import os
import csv
import random
from typing import List
import argparse


def split_data(label_info_path: str,
               output_dir: str,
               eval_ratio: float = 0.1,
               test_ratio: float = 0.1) -> None:
    """
    Split the data into training, evaluation, and testing sets based on unique sounds.

    1. Collect the unique sounds from label_info.csv.
    2. Randomly split those sounds into train, eval, and test sets.
    3. For each row in the label file:
       - If it has an event from the events_to_consider list, keep it.
       - Otherwise, label it as "NonEvent".
       - Assign the row to the train, eval, or test set according to the sound.
    4. Save to train.csv, eval.csv, and test.csv.

    Args:
    - label_info_path (str): The path to the label_info.csv file.
    - output_dir (str): The directory to save the split CSV files.
    - eval_ratio (float): The ratio of sounds to include in the evaluation set.
    - test_ratio (float): The ratio of sounds to include in the testing set.
    """

    # Read all rows of label_info
    with open(label_info_path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data_rows = list(reader)

    # Identify columns of interest
    # (Adjust "audio_file" to the actual column name where the sound identifier is located)
    try:
        sound_index = headers.index("audio_file")
    except ValueError:
        raise ValueError(
            "Expected 'audio_file' column in label_info.csv; please update accordingly."
        )

    label_index = headers.index("label_list")

    # Collect all unique sound identifiers
    unique_sounds = set(row[sound_index] for row in data_rows)
    unique_sounds = list(unique_sounds)
    random.shuffle(unique_sounds)

    # Determine sizes for splits (based on unique sounds count)
    total_sounds = len(unique_sounds)
    eval_size = int(total_sounds * eval_ratio)
    test_size = int(total_sounds * test_ratio)
    train_size = total_sounds - eval_size - test_size

    train_sounds = set(unique_sounds[:train_size])
    eval_sounds = set(unique_sounds[train_size:train_size + eval_size])
    test_sounds = set(unique_sounds[train_size + eval_size:])

    # Define events we want to keep as events (others become non-events)
    events_to_consider = [
        "Phonological Addition",
        "Phonological Deletion",
        "Phonological Substitution",
        "Lexical Substitution"
    ]

    # Prepare containers for final splits
    train_samples, eval_samples, test_samples = [], [], []

    for row in data_rows:
        # If row's label is not in events_to_consider, mark it as "NonEvent"
        if not any(event in row[label_index] for event in events_to_consider):
            row[label_index] = "NonEvent"

        # Assign row to the correct split based on its sound
        audio_file = row[sound_index]
        if audio_file in train_sounds:
            train_samples.append(row)
        elif audio_file in eval_sounds:
            eval_samples.append(row)
        else:
            test_samples.append(row)

    # Print out summary
    print(f"Total unique sounds: {total_sounds}")
    print(f"  Training sounds: {len(train_sounds)}")
    print(f"  Evaluation sounds: {len(eval_sounds)}")
    print(f"  Testing sounds: {len(test_sounds)}")
    print(f"\nFinal row counts:")
    print(f"  Training set: {len(train_samples)}")
    print(f"  Evaluation set: {len(eval_samples)}")
    print(f"  Testing set: {len(test_samples)}")

    # Save the splits
    save_split_csv(headers, train_samples, os.path.join(output_dir, "train.csv"))
    save_split_csv(headers, eval_samples, os.path.join(output_dir, "eval.csv"))
    save_split_csv(headers, test_samples, os.path.join(output_dir, "test.csv"))


def save_split_csv(headers: List[str],
                   samples: List[List[str]],
                   output_path: str) -> None:
    """
    Save the split samples to a CSV file.

    Args:
    - headers (List[str]): The header row for the CSV file.
    - samples (List[List[str]]): The samples to save.
    - output_path (str): The path to save the CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, eval, and test sets by unique sound")
    parser.add_argument("--label_info_path", type=str, required=True,
                        help="Path to the label info CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the split data")
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                        help="Ratio of sounds to include in the evaluation set")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio of sounds to include in the testing set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)

    split_data(
        label_info_path=args.label_info_path,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio
    )