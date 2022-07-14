mlwhatif
================================

[![mlinspect](https://img.shields.io/badge/❓-mlwhatif-green)](https://github.com/stefan-grafberger/mlwhatif)
[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://github.com/stefan-grafberger/mlwhatif/blob/master/LICENSE)
[![Build Status](https://github.com/stefan-grafberger/mlwhatif/actions/workflows/build.yml/badge.svg)](https://github.com/stefan-grafberger/mlwhatif/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/stefan-grafberger/mlwhatif/branch/main/graph/badge.svg?token=KTMNPBV1ZZ)](https://codecov.io/gh/stefan-grafberger/mlwhatif)

Data-Centric What-If Analysis for Native Machine Learning Pipelines 

## Run mlwhatif locally

Prerequisite: Python 3.9

1. Clone this repository
2. Set up the environment

	`cd mlwhatif` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>

3. If you want to use the visualisation functions we provide, install graphviz which can not be installed via pip

    `Linux: ` `apt-get install graphviz` <br>
    `MAC OS: ` `brew install graphviz` <br>
	
4. Install pip dependencies 

    `pip install -e ."[dev]"` <br>

5. To ensure everything works, you can run the tests (without graphviz, the visualisation test will fail)

    `python setup.py test` <br>

## Notes
* For debugging in PyCharm, set the pytest flag `--no-cov` ([Link](https://stackoverflow.com/questions/34870962/how-to-debug-py-test-in-pycharm-when-coverage-is-enabled))
* If you want to see log output in PyCharm, you can also set the pytest flags `--log-cli-level=10 -s`. The `-s` is needed because otherwise pytest breaks the stdout capturing.

## License
This library is licensed under the Apache 2.0 License.
