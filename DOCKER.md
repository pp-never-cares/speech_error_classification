# Project Setup and Docker Usage

This guide provides steps to compile the Docker image, upload it to a Docker registry, and download it for use.

## Directory Structure

Ensure your project directory has the following structure:

```
project/
│
├── Dockerfile
├── requirements.txt
├── experiments/
├── scripts/
├── src/
└── main.py
```

## Steps to Build the Docker Image

1. Open the terminal and navigate to the project directory.
2. Build the Docker image using the following command, with a custom tag name:

`docker build -t speech-error-ml .`

For MacOS user, it should be:

`docker buildx build --platform linux/amd64 -t speech-error-ml .`

## Steps to Upload the Docker Image to Docker Hub

1. Log in to Docker Hub using the following command:

`docker login`

2. Tag the Docker image with your Docker Hub username and repository name:

`docker tag speech-error-ml macarious/speech-error-ml`

3. Push the Docker image to Docker Hub:

`docker push macarious/speech-error-ml`

## Steps to Download the Docker Image

1. Load the Singularity module:

`module load singularity`

1. Pull the Docker image from Docker Hub using the following command:

`singularity pull speech-error-ml.sif docker://macarious/speech-error-ml`

This pulls the Docker image and converts it to a Singularity image `speech-error-ml_latest.sif`.

## Steps to Run the Docker Image

1. Use GPU from Northereastern's Discovery cluster:

(see https://github.com/SlangLab-NU/links/wiki/Working-with-sbatch-and-srun-on-the-cluster-with-GPU-nodes)

`srun --partition=gpu --nodes=1 --gres=gpu:t4:1 --time=08:00:00 --pty /bin/bash`

2. Load the Singularity module:

`module load singularity`

    For a safer process, check all versions of Singularity file:

`module unvail singularity`

    And choose the newest version( updated to 3.10.3 at Nov 2023 ):

`module load singularity/3.10.3`

3. Execute the image using the following command:

`singularity run --nv --bind /work/van-speech-nlp/hui.mac/sfused/data:/app/data,/work/van-speech-nlp/hui.mac/sfused/logs:/app/logs,/work/van-speech-nlp/hui.mac/sfused/experiments:/app/experiments,/work/van-speech-nlp/hui.mac/sfused/models:/app/models,/work/van-speech-nlp/hui.mac/sfused/checkpoints:/app/checkpoints --pwd /app /work/van-speech-nlp/hui.mac/sfused/speech-error-ml_latest.sif /bin/bash`

    "--bind" allows access to files under /src or /scripts too.

4. Run the Python script inside the container:

Convert audio files from .mp3 to .wav, and generate a list of all audio files:
```
bash scripts/process_audio_files.sh
```

Generate features, labels and split data using the list of audio files generated in the previous step:
```
bash scripts/generate_features.sh
bash scripts/split_data.sh
```

Train the model using an experiment configuration file (e.g., `experiments/exp_loss_0_binary_crossentropy.cfg`):
```
python3 src/training/main.py experiments/exp_loss_0_binary_crossentropy.cfg
```

Evaluate the model (e.g., `models\cluster-24-11-29\closs_cntrv1.00.keras`) against a specific audio file (e.g., `data\audio\ac003_2006-09-24.wav`):
```
bash scripts/evaluate_utterance.sh models/cluster-24-11-29/closs_cntrv1.00.keras data/audio/ac003_2006-09-24.wav
```

Read tensorboard logs (e.g., `logs/cluster-24-11-29/`):
```
python3 src/evaluation/read_tensorboard.py logs/cluster-24-11-29/


For simple baseline models, sandbox model running:

```
bash scripts/pipeline_downsample.sh
```

For simple models, running on resampled training dataset and un-resampled eval dataset:

```
bash scripts/pipeline_simple_models.sh
```

5. Clear cache if needed:

```
rm -rf /home/hui.mac/.cache/
rm -rf /home/hui.mac/.singularity/cache
```

6. Exit the container:

```
exit
```
