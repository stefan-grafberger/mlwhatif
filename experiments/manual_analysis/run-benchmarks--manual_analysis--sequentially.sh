#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
echo "Sequential execution";

data_loading_options=("fast_loading")


core_num="0-7"
echo "Cores to use: $core_num";

for model in "logistic_regression"
do
  for scenario in "data_cleaning" "data_corruption"
  do
    for dataset in "reviews"
    do
      for data_loading in "${data_loading_options[@]}"
      do
        for featurization in "featurization_1"
        do
          echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmark_manual_vs_automatic.py $scenario $dataset $data_loading $featurization $model"
          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmark_manual_vs_automatic.py "$scenario" "$dataset" "$data_loading" "$featurization" "$model"
        done
      done
    done
  done
done
