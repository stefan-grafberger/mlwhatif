#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
for scenario in "data_corruption"
do
  for dataset in "reviews"
  do
    for data_loading in "fast"
    do
      for featization in "fast"
      do
        for model in "logistinc_regression"
        do
          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--end-to-end.py "$scenario" "$dataset" "$data_loading" "$featization" "$model"
        done
      done
    done
  done
done
