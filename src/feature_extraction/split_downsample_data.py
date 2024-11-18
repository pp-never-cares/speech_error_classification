import os
import csv
import random
from typing import List
import argparse


def split_data(label_info_path: str, output_dir: str, eval_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
    """
    Split the data into training, evaluation, and testing sets, ensuring equal numbers of event and non-event samples.

    Args:
    - label_info_path (str): The path to the label_info_context.csv file.
    - output_dir (str): The directory to save the split CSV files.
    - eval_ratio (float): The ratio of samples to include in the evaluation set.
    - test_ratio (float): The ratio of samples to include in the testing set.
    """
    # Load the label_info file
    with open(label_info_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Get the headers
        label_info = list(reader)

    # Identify relevant events
    events_to_consider = ['Phonological Addition', 'Phonological Deletion', 'Phonological Substitution']
    label_index = headers.index('label_list')

    # Separate samples with and without events
    samples_with_events = [row for row in label_info if any(event in row[label_index] for event in events_to_consider)]
    samples_without_events = [row for row in label_info if all(event not in row[label_index] for event in events_to_consider)]

    # Shuffle the samples
    random.shuffle(samples_with_events)
    random.shuffle(samples_without_events)

    # Calculate the number of samples in each set
    num_event_samples = len(samples_with_events)
    num_eval_samples = int(num_event_samples * eval_ratio)
    num_test_samples = int(num_event_samples * test_ratio)
    num_train_samples = num_event_samples - num_eval_samples - num_test_samples

    # Split event samples into sets
    eval_samples_with_events = samples_with_events[:num_eval_samples]
    test_samples_with_events = samples_with_events[num_eval_samples:num_eval_samples + num_test_samples]
    train_samples_with_events = samples_with_events[num_eval_samples + num_test_samples:]

    # Select non-event samples to match the number of event samples in each set
    eval_samples_without_events = random.sample(samples_without_events, len(eval_samples_with_events))
    test_samples_without_events = random.sample(samples_without_events, len(test_samples_with_events))
    train_samples_without_events = random.sample(samples_without_events, len(train_samples_with_events))

    # Combine event and non-event samples for each set
    eval_samples = eval_samples_with_events + eval_samples_without_events
    test_samples = test_samples_with_events + test_samples_without_events
    train_samples = train_samples_with_events + train_samples_without_events

    # Shuffle each set to ensure randomness
    random.shuffle(eval_samples)
    random.shuffle(test_samples)
    random.shuffle(train_samples)

    # Print the distribution of samples in each set
    print(f"Training set: {len(train_samples)} samples (events: {len(train_samples_with_events)}, non-events: {len(train_samples_without_events)})")
    print(f"Evaluation set: {len(eval_samples)} samples (events: {len(eval_samples_with_events)}, non-events: {len(eval_samples_without_events)})")
    print(f"Testing set: {len(test_samples)} samples (events: {len(test_samples_with_events)}, non-events: {len(test_samples_without_events)})")

    # Save the split sets
    save_split_csv(headers, train_samples, os.path.join(output_dir, 'train_downsample.csv'))
    save_split_csv(headers, eval_samples, os.path.join(output_dir, 'eval_downsample.csv'))
    save_split_csv(headers, test_samples, os.path.join(output_dir, 'test_downsample.csv'))


def save_split_csv(headers: List[str], samples: List[List[str]], output_path: str) -> None:
    """
    Save the split samples to a CSV file.

    Args:
    - headers (List[str]): The header row for the CSV file.
    - samples (List[List[str]]): The samples to save.
    - output_path (str): The path to save the CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # Write the headers
        writer.writerows(samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split downsampled data into train, eval, and test sets with balanced classes.")
    parser.add_argument('--label_info_path', type=str, required=True, help="Path to the label_downsampled.csv file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the split data.")
    parser.add_argument('--eval_ratio', type=float, default=0.1, help="Ratio of eval data.")
    parser.add_argument('--test_ratio', type=float, default=0.1, help="Ratio of test data.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    label_info_path = args.label_info_path
    output_dir = args.output_dir
    eval_ratio = args.eval_ratio
    test_ratio = args.test_ratio
    seed = args.seed

    random.seed(seed)

    split_data(label_info_path, output_dir, eval_ratio, test_ratio)
