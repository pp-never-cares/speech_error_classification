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
EVAL_RATIO=0.3
TEST_RATIO=0.3
TRAIN_CSV_PATH="$LABEL_INFO_DIR/train.csv"
EVAL_CSV_PATH="$LABEL_INFO_DIR/eval.csv"
TEST_CSV_PATH="$LABEL_INFO_DIR/test.csv"
EPOCHS=50
BATCH_SIZE=64
FAILURE_LOG_DIR="${LABEL_INFO_DIR}/failure_log.log"


CSV_DIR="data/metadata/"
OUTPUT_DIR="data/metadata/"
CONTRIVE_RATIO=0.5
SEED=42

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
# python3 src/feature_extraction/generate_features.py --wav_list $WAVE_LIST --wav_dir $WAVE_DIR --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

# # Generate labels
# echo "Generating labels"
# if [ ! -d $LABEL_DIR ]; then
#     mkdir -p $LABEL_DIR
# fi
# if [ ! -f $FAILURE_LOG_DIR ]; then
#     touch $FAILURE_LOG_DIR
#     echo "Created empty log file at $FAILURE_LOG_DIR."
# fi
# python3 src/feature_extraction/generate_labels.py --annotations_path $ANNOTATIONS_PATH --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --label_dir $LABEL_DIR --label_info_dir $LABEL_INFO_DIR --feature_config $FEATURE_CONFIG --failure_log_dir $FAILURE_LOG_DIR --n_process $PROCESS_NUM

# # # Split data
# echo "Splitting data into train, eval, and test sets according to the auio list"
# python3 src/feature_extraction/split_audio_data.py --label_info_path $LABEL_INFO_PATH --output_dir $OUTPUT_DIR --eval_ratio $EVAL_RATIO --test_ratio $TEST_RATIO

# # # # Split data
# # # echo "Splitting data into train, eval, and test sets according to the autio channel data"
# # # python3 src/feature_extraction/split_channel_data.py --label_info_path $LABEL_INFO_PATH --output_dir $OUTPUT_DIR --eval_ratio $EVAL_RATIO --test_ratio $TEST_RATIO

# # Train model
# echo "Training baseline model with baseline setting"
# python3 src/training/main.py experiments/baseline.cfg


# # Create output directory if it doesn't exist
# if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
# mkdir -p "$(dirname "OUTPUT_DIR")"
# fi

# echo "Creating contrived datasets with balanced event and non-event samples..."

# python src/feature_extraction/create_contrive_set.py \
# --csv_dir "$CSV_DIR" \
# --output_dir "$OUTPUT_DIR" \
# --ratio "$CONTRIVE_RATIO" \
# --seed "$SEED"

# if [ $? -ne 0 ]; then
# echo "Error: create_contrive_set.py failed."
# exit 1
# fi

# echo "Contrived datasets created successfully."
# echo "Contrived data is located in: $OUTPUT_DIR"

# train model with contrived data. 

echo "Training baseline model with contrived setting"
python3 src/training/main.py experiments/baseline_contrive_0.50.cfg

echo "Training baseline model with closs_cntrv0.50 setting"
python3 src/training/main.py experiments/closs_cntrv0.50.cfg


echo "Training baseline model with closs_cntrv0.50_fweight0 setting"
python3 src/training/main.py experiments/closs_cntrv0.50_fweight0.0.cfg

echo "Training baseline model with closs_cntrv0.20_fweight0.25 setting"
python3 src/training/main.py experiments/closs_cntrv0.20_fweight0.25.cfg

echo "Training baseline model with closs_cntrv0.20_fweight0.50 setting"
python3 src/training/main.py experiments/closs_cntrv0.20_fweight0.50.cfg

echo "Training baseline model with closs_cntrv0.20_fweight0.75 setting"
python3 src/training/main.py experiments/closs_cntrv0.20_fweight0.75.cfg

echo "Training baseline model with closs_cntrv0.20_uweight0 setting"
python3 src/training/main.py experiments/closs_cntrv0.20_uweight0.0.cfg


echo "Training done."



