# Sound Event Detection for Speech Error Detection

To run the code, you need to install the following libraries:

- python=3.10
- numpy<2
- librosa
- soxr
- scipy
- soundfile
- audioread
- pandas
- tensorflow
- keras

Save the audio files (.wav) in the `data/audio` folder.

The dataset (.csv) should be saved in the `data/metadata` folder. The dataset should have the following columns:

- `file`: the name of the audio file
- `label`: the label of the speech error
- `start`: the start time of the speech error
- `end`: the end time of the speech error

The WhisperX transcript files (.csv) should be saved in the `data/whisperX` folder. The transcript files should have the following columns:

- `start`: the start time of a speech segment
- `end`: the end time of a speech segment
- `text`: the text of the speech segment

To load the docker image on the Northeastern Discovery cluster and running the model, refer to the [DOCKER.md](DOCKER.md) file.

## Folder Structure

The code is organized with the following main folders and files:

```
├── checkpoints/
├── data/
│   ├── audio/
│   ├── features/
│   ├── labels/
│   ├── metadata/
│   ├── visualizations/
│   ├── whisperX/
│   └── whisperX_word/
├── experiments/
├── logs/
├── models/
├── predictions/
├── scripts/
│   ├── create_contrive_set.sh
│   ├── evaluate_utterance.sh
│   ├── generate_features.sh
│   ├── generate_labels.sh
│   ├── process_audio_files.sh
│   ├── split_data.sh
│   └── train_model.sh
├── src/
│   ├── audio_processing/
│   │   ├── convert_mp3_to_wav.py
│   │   ├── generate_audio_list.py
│   │   └── visualize_audio.py
│   ├── evaluation/
│   │   ├── evaluate_utterance.py
│   │   ├── label_comparison.py
│   │   ├── model_prediction.py
│   │   ├── read_tensorboard.py
│   │   └── transcript_annotation.py
│   ├── feature_extraction/
│   │   ├── create_contrive_set.py
│   │   ├── feature.cfg
│   │   ├── generate_features.py
│   │   ├── generate_labels.py
│   │   └── split_data.py
│   ├── test/
│   │   ├── custom_loss_test.py
│   │   └── validate_labels.py
│   └── training/
│       ├── attention.py
│       ├── custom_data_generator.py
│       ├── custom_error_rate_metric.py
│       ├── custom_f1_score.py
│       ├── custom_frame_level_loss.py
│       ├── data_utils.py
│       ├── main.py
│       ├── model_trainer.py
│       ├── model_utils.py
│       └── parse_config.py
├── DOCKER.md
├── Dockerfile
├── environment.yml
├── LICENSE
├── README.md
├── requirements.txt
└── sbatch_sfused.sh
```

## Descriptions

### **Folders**

#### `checkpoints/`

Contains the model checkpoints saved during training (`.keras` files).

#### `data/`

- `audio/`: Stores audio files (`.wav`).
- `features/`: Stores features extracted from the audio files (`.npy`).
- `labels/`: Stores labels for the audio files (`.npy`).
- `metadata/`: Stores metadata for the dataset (`.csv`).
- `visualizations/`: Stores waveforms and spectrograms (`.png`).
- `whisperX/`: Contains WhisperX transcript files (`.csv`) with start, end, and text columns.
- `whisperX_word/`: Contains word- and segment-level WhisperX transcripts (`.json`).

#### `experiments/`

Stores experiment configuration files (`.cfg`).

#### `logs/`

Stores logs of training processes, including TensorBoard logs (`.csv`, `.json`, `.log`).

#### `models/`

Contains trained models (`.keras`).

#### `predictions/`

Stores model predictions on the test set (`.json`).

#### `scripts/`

Shell scripts for data preparation, feature generation, and model training:

- `create_contrive_set.sh`: Creates a contrived dataset with only utterances containing speech errors.
- `evaluate_utterance.sh`: __Evaluates the model on a specific audio file.__
- `generate_features.sh`: Extracts features from audio files.
- `generate_labels.sh`: Generates labels from metadata.
- `process_audio_files.sh`: Converts audio files from `.mp3` to `.wav`.
- `split_data.sh`: Splits data into training, validation, and test sets.
- `train_model.sh`: __Trains the model using specified configurations.__

#### `src/`

Source code organized into subfolders:

##### `audio_processing/`

- `convert_mp3_to_wav.py`: Converts audio files from `.mp3` to `.wav`.
- `generate_audio_list.py`: Generates a list of all audio files and saves it in the `data/metadata` folder.
- `visualize_audio.py`: Generates waveforms and spectrograms for audio files and saves them in the `data/visualizations` folder.

##### `evaluation/`

- `evaluate_utterance.py`: Evaluates the model on a specific audio file.
- `label_comparison.py`: Compares predicted labels with ground truth and calculates evaluation metrics.
- `model_prediction.py`: Generates predictions for audio files using the trained model.
- `read_tensorboard.py`: __Reads TensorBoard logs for analysis.__
- `transcript_annotation.py`: Annotates WhisperX transcript files with predicted speech errors.

##### `feature_extraction/`

- `create_contrive_set.py`: Creates a contrived dataset with only utterances containing speech errors.
- `feature.cfg`: Configuration file for feature extraction.
- `generate_features.py`: Extracts features from audio files and saves them in `data/features`.
- `generate_labels.py`: Generates labels from metadata and saves them in `data/labels`.
- `split_data.py`: Splits data into training, validation, and test sets.

##### `test/`

- `custom_loss_test.py`: Tests custom loss functions for the model.
- `validate_labels.py`: Validates the correctness of generated labels.

##### `training/`

- `attention.py`: Defines a custom Keras layer for attention mechanisms.
- `custom_data_generator.py`: Implements a custom data generator for model training.
- `custom_error_rate_metric.py`: Defines a custom error rate metric (not currently used).
- `custom_f1_score.py`: Implements a custom F1 score metric.
- `custom_frame_level_loss.py`: Implements a custom loss function for frame-level predictions.
- `data_utils.py`: Utility functions for loading and processing data from `.csv` and `.npy` files.
- `main.py`: __Main script for training the model.__
- `model_trainer.py`: Implements a class for training the model with k-fold cross-validation.
- `model_utils.py`: Utility functions for building and training the Keras model.
- `parse_config.py`: Utility functions for parsing experiment configuration files.

---

### **Files**

- **DOCKER.md**: Instructions to load the Docker image and run the model on the Northeastern Discovery cluster.
- **Dockerfile**: Dockerfile for building the Docker image.
- **environment.yml**: Conda environment configuration file.
- **LICENSE**: MIT License for the project.
- **README.md**: Project documentation (this file).
- **requirements.txt**: List of required Python libraries.
- **sbatch_sfused.sh**: Script to run the model on the Northeastern Discovery cluster using GPU nodes.


## Run it with Annoconda 

## Running the Project with Anaconda

This guide explains how to set up and run the Speech Error Detection project in a local environment using [Anaconda](https://www.anaconda.com/).

#### Step 1: Clone the Repository

```bash
git clone https://github.com/pp-never-cares/speech_error_classification
cd speech_error_classification
```

#### Step 2: Create and Activate the Conda Environment

If an `environment.yml` file is available, run:

```bash
conda env create -f environment.yml
conda activate speech-error
```

If not, create the environment manually:

```bash
conda create -n speech-error python=3.10
conda activate speech-error
```

Then install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Required packages include:

* numpy < 2
* librosa
* soxr
* scipy
* soundfile
* audioread
* pandas
* tensorflow
* keras

#### Step 3: Prepare the Data

Ensure your data is organized as follows:

* Audio files: `data/audio/*.wav`
* Metadata CSV: `data/metadata/*.csv`
* WhisperX transcripts: `data/whisperX/*.csv`

Each metadata CSV should contain:

* `file`, `label`, `start`, `end`

Each transcript CSV should contain:

* `start`, `end`, `text`

#### Step 4: Run Preprocessing Scripts

```bash
bash scripts/process_audio_files.sh
bash scripts/generate_features.sh
bash scripts/split_data.sh
```
If you want everythin to be done in one step, run 
```bash
bash scripts/pipeline.sh
```


#### Step 5: Train the Model

```bash
python src/training/main.py experiments/baseline_contrive_0.50.cfg
```

#### Step 6: Evaluate the Model

```bash
bash scripts/evaluate_utterance.sh models/YOUR_MODEL_FILE.keras data/audio/YOUR_AUDIO_FILE.wav
```

#### Step 7: Visualize Logs

```bash
python src/evaluation/read_tensorboard.py logs/YOUR_LOG_DIR/
```
