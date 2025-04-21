import os
import csv
import random
from typing import List, Dict
import argparse
from collections import defaultdict


def get_channel_name(audio_file: str) -> str:
    # Define how to extract channel (e.g., first two characters)
    return audio_file[:2]


def split_data(label_info_path: str,
               output_dir: str,
               eval_ratio: float = 0.1,
               test_ratio: float = 0.1) -> None:
    """
    Split the data into training, evaluation, and testing sets per channel (grouped by audio_file prefix).

    Args:
    - label_info_path (str): Path to label_info.csv.
    - output_dir (str): Directory to save the split CSVs.
    - eval_ratio (float): Ratio for eval set.
    - test_ratio (float): Ratio for test set.
    """

    # Read data
    with open(label_info_path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data_rows = list(reader)

    sound_index = headers.index("audio_file")
    label_index = headers.index("label_list")

    events_to_consider = [
        "Phonological Addition",
        "Phonological Deletion",
        "Phonological Substitution"
    ]

    # Group data by channel
    channel_data: Dict[str, List[List[str]]] = defaultdict(list)
    for row in data_rows:
        channel = get_channel_name(row[sound_index])
        channel_data[channel].append(row)

    # Final splits
    train_samples, eval_samples, test_samples = [], [], []

    for channel, rows in channel_data.items():
        unique_sounds = list({row[sound_index] for row in rows})
        random.shuffle(unique_sounds)

        total = len(unique_sounds)
        eval_size = int(total * eval_ratio)
        test_size = int(total * test_ratio)
        train_size = total - eval_size - test_size

        train_sounds = set(unique_sounds[:train_size])
        eval_sounds = set(unique_sounds[train_size:train_size + eval_size])
        test_sounds = set(unique_sounds[train_size + eval_size:])

        for row in rows:
            # Re-label non-events
            if not any(event in row[label_index] for event in events_to_consider):
                row[label_index] = "NonEvent"

            sound = row[sound_index]
            if sound in train_sounds:
                train_samples.append(row)
            elif sound in eval_sounds:
                eval_samples.append(row)
            else:
                test_samples.append(row)

    # Print stats
    print(f"Processed {len(channel_data)} channels.")
    print(f"Train rows: {len(train_samples)} | Eval rows: {len(eval_samples)} | Test rows: {len(test_samples)}")

    save_split_csv(headers, train_samples, os.path.join(output_dir, "train.csv"))
    save_split_csv(headers, eval_samples, os.path.join(output_dir, "eval.csv"))
    save_split_csv(headers, test_samples, os.path.join(output_dir, "test.csv"))


def save_split_csv(headers: List[str],
                   samples: List[List[str]],
                   output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data by channel into train, eval, and test sets")
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