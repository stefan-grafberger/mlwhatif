#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
echo "Sequential execution";

core_num="0-7"
echo "Cores to use: $core_num";

#for variant_count in "0" "2" "4" "6" "8" "10" "12" "14" "16"
for variant_count in "0" "4" "8" "12" "16"
do
  #for model in "logistic_regression" "xgboost" "neural_network"
  for model in "logistic_regression"
  do
  #  for scenario in "data_cleaning" "data_corruption" "feature_importance" "operator_impact"
    for scenario in "data_cleaning" "data_corruption"
#    for scenario in "data_corruption"
    do
  #    for dataset in "healthcare" "folktables" "cardio" "reviews"
      for dataset in "reviews_10x"
      do
        for data_loading in "fast_loading"
        do
  #        for featurization in "featurization_0" "featurization_1" "featurization_2" "featurization_3" "featurization_4"
          for featurization in "featurization_2"
          do
            for exec_strategy in "dfs" "bfs"
            do
              echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_memory.py $scenario $dataset $data_loading $featurization $model $exec_strategy $variant_count"
              PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_memory.py "$scenario" "$dataset" "$data_loading" "$featurization" "$model" "$exec_strategy" "$variant_count"
            done
          done
        done
      done
    done
  done
done
