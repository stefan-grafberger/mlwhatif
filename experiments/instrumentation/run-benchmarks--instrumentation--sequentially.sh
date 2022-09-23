#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
echo "Sequential execution";

core_num="0-7"
echo "Cores to use: $core_num";


for data_loading in "fast_loading" "slow_loading"
do
  echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $dataset $data_loading image image"
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_instrumentation.py sneakers "$data_loading" image image
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks_instrumentation.py sneakers "$data_loading" image image
done

for dataset in "healthcare" "folktables" "cardio" "reviews"
do
  for data_loading in
  do
    for featurization in "featurization_0" "featurization_1" "featurization_2" "featurization_3" "featurization_4"
    do
      for model in "logistic_regression" "xgboost" "neural_network"
      do
        echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $dataset $data_loading $featurization $model"
#        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_instrumentation.py "$dataset" "$data_loading" "$featurization" "$model"
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks_instrumentation.py "$dataset" "$data_loading" "$featurization" "$model"
      done
    done
  done
done
