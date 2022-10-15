mlmq
================================

[![mlmq](https://img.shields.io/badge/❓-mlmq-green)](https://github.com/anonymous-52200/mlmq)
[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://github.com/anonymous-52200/mlmq/blob/master/LICENSE)
[![Build Status](https://github.com/anonymous-52200/mlmq/actions/workflows/build.yml/badge.svg)](https://github.com/anonymous-52200/mlmq/actions/workflows/build.yml)

Data-Centric What-If Analysis for Native Machine Learning Pipelines.

This project uses the [mlinspect](https://github.com/stefan-grafberger/mlinspect) project as a foundation, mainly for its plan extraction from native ML pipelines.

## Run mlmq locally

Prerequisite: Python 3.9

1. Clone this repository
2. Set up the environment

	`cd mlmq` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>

3. If you want to use the visualisation functions we provide, install graphviz which can not be installed via pip

    `Linux: ` `apt-get install graphviz` <br>
    `MAC OS: ` `brew install graphviz` <br>
	
4. Install pip dependencies 

    `pip install -e ."[dev]"` <br>

5. To ensure everything works, you can run the tests (without graphviz, the visualisation test will fail)

    `python setup.py test` <br>

## Detailed Example
We prepared a [demo notebook](demo/feature_overview/feature_overview.ipynb) to showcase mlmq and its features.

## Notes
* For debugging in PyCharm, set the pytest flag `--no-cov` ([Link](https://stackoverflow.com/questions/34870962/how-to-debug-py-test-in-pycharm-when-coverage-is-enabled))
* If you want to see log output in PyCharm, you can also set the pytest flags `--log-cli-level=10 -s`. The `-s` is needed because otherwise pytest breaks the stdout capturing.

## License
This library is licensed under the Apache 2.0 License.
