#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
if [[ $# -eq 0 ]] ; then
    echo 'No argument supplied'
    exit 1
fi

echo "Variant name from 1-8: $1";
start_core_index=$((($1 % 8) * 32))
end_core_index=$((start_core_index + 7))
core_num="$start_core_index"-"$end_core_index"

if [[ "$(($1 % 4))" == "0" ]]
then
  datasets=("healthcare")
elif [[ "$(($1 % 4))" == "1" ]]
then
  datasets=("reviews")
elif [[ "$(($1 % 4))" == "2" ]]
then
  datasets=("folktables")
elif [[ "$(($1 % 4))" == "3" ]]
then
  datasets=("cardio")
fi

if (( $1 < 4 ))
then
  data_loading_options=("fast_loading")
else
  data_loading_options=("slow_loading")
fi

echo "Cores to use: $core_num";
for scenario in "data_corruption" "feature_importance" "data_cleaning" "operator_impact"
do
  for dataset in "${datasets[@]}"
  do
    for data_loading in "${data_loading_options[@]}"
    do
      for featurization in "featurization_0" "featurization_1" "featurization_2" "featurization_3" "featurization_4"
      do
        for model in "logistic_regression" "xgboost" "neural_network"
        do
          echo "taskset -c $core_num PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks_end_to_end.py $scenario $dataset $data_loading $featurization $model"
          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" "$dataset" "$data_loading" "$featurization" "$model"
        done
      done
    done
  done
done

if [[ "$(($1 % 4))" == "0" ]]
then
  for scenario in "data_corruption" "data_cleaning"
  do
    for data_loading in "${data_loading_options[@]}"
    do
      echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $scenario $dataset $data_loading image image"
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" sneakers "$data_loading" image image
    done
  done
fi
