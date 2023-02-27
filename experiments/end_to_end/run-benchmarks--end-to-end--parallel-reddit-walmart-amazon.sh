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

if (( (($1 % 4)) < 2 ))
then
  scenarios=("feature_importance" "data_corruption")
else
  scenarios=("operator_impact" "data_cleaning")
fi

data_loading_options=("fast_loading")

echo "Cores to use: $core_num";

if [[ "$(($1 % 2))" == "0" ]]
then
  for scenario in "${scenarios[@]}"
  do
    for data_loading in "${data_loading_options[@]}"
    do
      echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $scenario reddit $data_loading reddit reddit"
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" reddit "$data_loading" reddit reddit
#      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks_end_to_end.py "$scenario" reddit "$data_loading" reddit reddit
    done
  done
fi

if [[ "$(($1 % 2))" == "1" ]]
then
  for scenario in "${scenarios[@]}"
  do
    for data_loading in "${data_loading_options[@]}"
    do
      echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c $core_num python3.9 benchmarks_end_to_end.py $scenario walmart_amazon $data_loading walmart_amazon walmart_amazon"
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c "$core_num" python3.9 benchmarks_end_to_end.py "$scenario" walmart_amazon "$data_loading" walmart_amazon walmart_amazon
#      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks_end_to_end.py "$scenario" walmart_amazon "$data_loading" walmart_amazon walmart_amazon
    done
  done
fi
