#!/bin/bash

# Define Python interpreter path
python_cmd=$(which python3)

AUDIO_DIR=data/audio
LIST_OUTPUT=data/metadata/wav_list.lst
SAMPLING_RATE=16000
PROCESS_NUM=4
WAVE_LIST=$LIST_OUTPUT
WAVE_DIR=data/audio
TRANSCRIPT_DIR=data/whisperX
FEATURE_DIR=data/features
LABEL_DIR=data/labels
FEATURE_CONFIG=src/feature_extraction/feature.cfg
ANNOTATIONS_PATH=data/metadata/dataset.csv
LABEL_INFO_DIR=data/metadata
LABEL_INFO_PATH="$LABEL_INFO_DIR/label_info.csv"
OUTPUT_DIR=$LABEL_INFO_DIR
EVAL_RATIO=0.1
TEST_RATIO=0.1
TRAIN_CSV_PATH="$LABEL_INFO_DIR/train.csv"
EVAL_CSV_PATH="$LABEL_INFO_DIR/eval.csv"
TEST_CSV_PATH="$LABEL_INFO_DIR/test.csv"
EPOCHS=50
BATCH_SIZE=64
EXPERIMENT_CONFIG_PATH=experiments/default.cfg


#define the downsampled data directory
DOWNSAMPLED_FEATURE_DIR=data/downsampled_features
DOWNSAMPLED_LABEL_DIR=data/downsampled_labels
DOWNSAMPLED_LABEL_INFO_PATH="$LABEL_INFO_DIR/label_downsampled.csv"
CONTEXTUAL_FEATURE_DIR=data/contextual_features
CONTEXTUAL_LABEL_DIR=data/contextual_labels
LABEL_INFO_CONTEXT_PATH="$LABEL_INFO_DIR/label_info_context.csv"
WINDOW_SIZE=5

# Define simple model train/eval data paths
TRAIN_DATA_PATH="data/metadata/train_downsampled.csv"
EVAL_DATA_PATH="data/metadata/eval_downsample.csv"
TEST_DATA_PATH="data/metadata/test_downsample.csv"






# # Convert mp3 to wav
# echo "Converting mp3 to wav"
# python src/audio_processing/convert_mp3_to_wav.py --audio_dir $AUDIO_DIR --output $AUDIO_DIR --sample_rate $SAMPLING_RATE

# # Generate audio list
# echo "Generating audio list"
# python src/audio_processing/generate_audio_list.py --audio_dir $AUDIO_DIR --output $LIST_OUTPUT

# # Generate features
# echo "Extracting features"
# if [ ! -d $FEATURE_DIR ]; then
#     mkdir -p $FEATURE_DIR
# fi
# python src/feature_extraction/generate_features.py --wav_list $WAVE_LIST --wav_dir $WAVE_DIR --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

# # Generate labels
# echo "Generating labels"
# if [ ! -d $LABEL_DIR ]; then
#     mkdir -p $LABEL_DIR
# fi
# python src/feature_extraction/generate_labels.py --annotations_path $ANNOTATIONS_PATH --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --label_dir $LABEL_DIR --label_info_dir $LABEL_INFO_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

# Add contextual features and labels
echo "Adding contextual features and labels"
if [ ! -d $CONTEXTUAL_FEATURE_DIR ]; then
    mkdir -p $CONTEXTUAL_FEATURE_DIR
fi
if [ ! -d $CONTEXTUAL_LABEL_DIR ]; then
    mkdir -p $CONTEXTUAL_LABEL_DIR
fi

python src/feature_extraction/add_contextual_features.py \
    --label_info_path $LABEL_INFO_PATH \
    --output_path $LABEL_INFO_CONTEXT_PATH \
    --feature_dir $FEATURE_DIR \
    --contextual_feature_dir $CONTEXTUAL_FEATURE_DIR \
    --contextual_label_dir $CONTEXTUAL_LABEL_DIR \
    --window_size $WINDOW_SIZE


# Downsample non-error data to address class imbalance using the contextual features
echo "Downsampling data for balance"
if [ ! -d $CONTEXTUAL_FEATURE_DIR ]; then
    mkdir -p $CONTEXTUAL_FEATURE_DIR
fi
if [ ! -d $CONTEXTUAL_LABEL_DIR ]; then
    mkdir -p $CONTEXTUAL_LABEL_DIR
fi
python src/feature_extraction/downsampling.py \
    --label_info_path $LABEL_INFO_CONTEXT_PATH \
    --output_path $DOWNSAMPLED_LABEL_INFO_PATH \
    --contextual_feature_dir $CONTEXTUAL_FEATURE_DIR \
    --label_dir $CONTEXTUAL_LABEL_DIR \
    --downsampled_feature_dir $DOWNSAMPLED_FEATURE_DIR \
    --downsampled_label_dir $DOWNSAMPLED_LABEL_DIR \



# Split data into train, eval, and test sets
echo "Splitting added contextual feature data into train, eval, and test sets"
python src/feature_extraction/split_downsample_data.py --label_info_path $DOWNSAMPLED_LABEL_INFO_PATH --output_dir $OUTPUT_DIR --eval_ratio $EVAL_RATIO --test_ratio $TEST_RATIO


echo "Training Logistic Regression model"
python src/downsample/LR_train_and_evaluate.py \
    --train_csv_path $TRAIN_DATA_PATH \
    --eval_csv_path $EVAL_DATA_PATH \
    --test_csv_path $TEST_DATA_PATH \
    # --epochs $EPOCHS \
    # --batch_size $BATCH_SIZE \
    # --config_path $EXPERIMENT_CONFIG_PATH \
    # --output_model_path models/best_logistic_model

echo "Training Support Vector Machine model"
python src/downsample/SVM_train_and_evaluate.py \
    --train_csv_path $TRAIN_DATA_PATH \
    --eval_csv_path $EVAL_DATA_PATH \
    --test_csv_path $TEST_DATA_PATH \
    # --epochs $EPOCHS \
    # --batch_size $BATCH_SIZE \
    # --config_path $EXPERIMENT_CONFIG_PATH \
    # --output_model_path models/best_rf_model.joblib


echo "Training Random Forest model"
python src/downsample/RF_train_and_evaluate.py \
    --train_csv_path $TRAIN_DATA_PATH \
    --eval_csv_path $EVAL_DATA_PATH \
    --test_csv_path $TEST_DATA_PATH \
    # --epochs $EPOCHS \
    # --batch_size $BATCH_SIZE \
    # --config_path $EXPERIMENT_CONFIG_PATH \
    # --output_model_path models/best_svm_model.joblib



echo "Pipeline execution completed."


