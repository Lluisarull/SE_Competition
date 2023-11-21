#!/bin/bash

# You can run this script from the command line using:
# ./run_pipeline.sh <start_date> <end_date> <raw_data_path> <master_gen_file> <master_load_file> <clean_data_file> <model_file> <predictions_file>
# For example:
# sh scripts/run_pipeline.sh 2022-01-01 2023-01-01 data/ data/master_gen.csv data/master_load.csv data/clean/data.csv data/clean/train.csv data/clean/test.csv models/model_dictionary.pickle predictions/predictions.json

# Get command line arguments
start_date="$1"
end_date="$2"
raw_data_path="$3"
master_gen_file="$4"
master_load_file="$5"
clean_data_file="$6"
train_file="$7"
test_file="$8"
model_file="$9"
predictions_file="${10}"

# Run data_ingestion.py
echo "Starting data ingestion..."
python src/data_ingestion.py --start_time="$start_date" --end_time="$end_date" --output_path="$raw_data_path"

# Run data_processing.py
echo "Starting data processing..."
python src/data_processing.py --input_gen_file="$master_gen_file" --input_load_file="$master_load_file" --output_data_file="$clean_data_file" --train_output="$train_file" --test_output="$test_file"

# Run model_training.py
echo "Starting model training..."
python src/model_training.py --input_file="$clean_data_file" --model_file="$model_file"

# Run model_prediction.py
echo "Starting prediction..."
python src/model_prediction.py --model_file="$model_file" --output_file="$predictions_file"

echo "Pipeline completed."
