#!/bin/bash
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues
for variant_count in 2 4 6 8
do
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py common_subexpression_elimination ideal "$variant_count" 2000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py common_subexpression_elimination average "$variant_count" 2000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py common_subexpression_elimination worst "$variant_count" 2000
#
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up ideal "$variant_count" 2000
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up average "$variant_count" 2000
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up worst_wo_safety "$variant_count" 2000
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up worst_w_safety "$variant_count" 2000
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up worst_safety_too_defensive "$variant_count" 2000
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py operator_deletion_filter_push_up worst_case_only_some_filters_worth_pushing_up "$variant_count" 2000
#
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py filter_addition_push_up ideal "$variant_count" 5000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py filter_addition_push_up average "$variant_count" 5000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py filter_addition_push_up worst_wo_original "$variant_count" 5000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py filter_addition_push_up worst_w_original "$variant_count" 5000
#
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up ideal "$variant_count" 2000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up average "$variant_count" 2000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up worst "$variant_count" 2000
#
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse ideal "$variant_count" 10000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse average "$variant_count" 10000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse worst_w_safety "$variant_count" 10000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse worst_wo_safety "$variant_count" 10000
#  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py udf_split_and_reuse worst_constant "$variant_count" 10000
done
