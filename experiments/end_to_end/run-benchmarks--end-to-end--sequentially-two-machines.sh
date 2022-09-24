#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
echo "Sequential execution";

if [[ $# -eq 0 ]] ; then
    echo 'No argument supplied'
    exit 1
fi

echo "Machine index from 0-1: $1";

if (( (($1)) < 1 ))
then
  data_loading_options=("fast_loading")
else
  data_loading_options=("slow_loading")
fi

core_num="0-7"
echo "Cores to use: $core_num";

for scenario in "data_cleaning" "data_corruption"
do
  for data_loading in "${data_loading_options[@]}"
  do
    echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $scenario $dataset $data_loading image image"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" sneakers "$data_loading" image image
  done
done

for model in "logistic_regression" "xgboost" "neural_network"
do
  for scenario in "data_cleaning" "data_corruption" "feature_importance" "operator_impact"
  do
    for dataset in "healthcare" "folktables" "cardio" "reviews"
    do
      for data_loading in "${data_loading_options[@]}"
      do
        for featurization in "featurization_0" "featurization_1" "featurization_2" "featurization_3" "featurization_4"
        do
          echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $scenario $dataset $data_loading $featurization $model"
          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" "$dataset" "$data_loading" "$featurization" "$model"
        done
      done
    done
  done
done
