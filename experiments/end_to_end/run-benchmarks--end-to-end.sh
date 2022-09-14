#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
for scenario in "data_corruption" "feature_importance" "data_cleaning" "operator_impact"
do
  for dataset in "reviews"
  do
    for data_loading in "fast_loading" "slow_loading"
    do
      for featurization in "fast"
      do
        for model in "logistic_regression" "xgboost" "neural_network"
        do
          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks_end_to_end.py "$scenario" "$dataset" "$data_loading" "$featurization" "$model"
        done
      done
    done
  done
done
