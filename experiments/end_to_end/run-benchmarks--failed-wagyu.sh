PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c 32-39 python3.9 benchmarks_end_to_end.py data_corruption folktables slow_loading featurization_1 logistic_regression && \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c 32-39 python3.9 benchmarks_end_to_end.py data_corruption cardio slow_loading featurization_0 xgboost && \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python taskset -c 32-39 python3.9 benchmarks_end_to_end.py data_cleaning cardio slow_loading featurization_0 logistic_regression && \
