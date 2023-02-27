#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
if [[ $# -eq 0 ]] ; then
    echo 'No argument supplied'
    exit 1
fi

echo "Variant name from 0-3: $1";
start_core_index=$((($1 % 4) * 64))
end_core_index=$((start_core_index + 7))
core_num="$start_core_index"-"$end_core_index"

if [[ "$(($1 % 4))" == "0" ]]
then
  scenarios=("data_corruption" "feature_importance")
  datasets=("folktables_1x" "folktables_5x" "folktables_10x")
elif [[ "$(($1 % 4))" == "1" ]]
then
  scenarios=("data_corruption" "feature_importance")
  datasets=("reviews_1x" "reviews_5x" "reviews_10x")
elif [[ "$(($1 % 4))" == "2" ]]
then
  scenarios=("data_cleaning" "operator_impact")
  datasets=("folktables_1x" "folktables_5x" "folktables_10x")
elif [[ "$(($1 % 4))" == "3" ]]
then
  scenarios=("data_cleaning" "operator_impact")
  datasets=("reviews_1x" "reviews_5x" "reviews_10x")
fi

featurizations=("featurization_2")


echo "Cores to use: $core_num";
for scenario in "${scenarios[@]}"
do
  for dataset in "${datasets[@]}"
  do
    for data_loading in "fast_loading"
    do
      for featurization in "${featurizations[@]}"
      do
#        for model in "logistic_regression" "xgboost" "neural_network"
        for model in "xgboost"
        do
          echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $scenario $dataset $data_loading $featurization $model 3"
          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" "$dataset" "$data_loading" "$featurization" "$model" 3
#          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks_end_to_end.py "$scenario" "$dataset" "$data_loading" "$featurization" "$model"
        done
      done
    done
  done
done
