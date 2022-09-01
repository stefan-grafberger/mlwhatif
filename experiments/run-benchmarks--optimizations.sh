#!/bin/bash

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required on our Azure machine due to some dependency issues

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up ideal 5 1000
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up average 5 1000
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3.9 benchmarks--optimization.py projection_push_up worst 5 1000
