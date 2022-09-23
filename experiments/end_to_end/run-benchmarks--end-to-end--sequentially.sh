#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
echo "Sequential execution";

core_num="0-7"
echo "Cores to use: $core_num";

for scenario in "data_corruption" "data_cleaning"
do
  for data_loading in "fast_loading" "slow_loading"
  do
    echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $scenario $dataset $data_loading image image"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" sneakers "$data_loading" image image
  done
done

for scenario in "feature_importance" "operator_impact" "data_corruption" "data_cleaning"
do
  for dataset in "healthcare" "folktables" "cardio" "reviews"
  do
    for data_loading in
    do
      for featurization in "featurization_0" "featurization_1" "featurization_2" "featurization_3" "featurization_4"
      do
        for model in "logistic_regression" "xgboost" "neural_network"
        do
          echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $scenario $dataset $data_loading $featurization $model"
          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" "$dataset" "$data_loading" "$featurization" "$model"
        done
      done
    done
  done
done
