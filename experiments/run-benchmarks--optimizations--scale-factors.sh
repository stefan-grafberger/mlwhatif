#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
for scale_factor in 0.5 1.0 # 1.5 2.0 2.5 3.0
do
  for variant_count in 4 # 2 6 8 10
  do
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py common_subexpression_elimination ideal "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py common_subexpression_elimination average "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py common_subexpression_elimination worst "$variant_count" "$scale_factor"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up ideal "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up average "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up worst_wo_safety "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up worst_w_safety "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up worst_safety_too_defensive "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up worst_case_only_some_filters_worth_pushing_up "$variant_count" "$scale_factor"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py filter_addition_push_up ideal "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py filter_addition_push_up average "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py filter_addition_push_up worst_wo_original "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py filter_addition_push_up worst_w_original "$variant_count" "$scale_factor"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up ideal "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up average "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up worst "$variant_count" "$scale_factor"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse ideal "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse average "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse worst_w_safety "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse worst_wo_safety "$variant_count" "$scale_factor"
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse worst_constant "$variant_count" "$scale_factor"
  done
done
