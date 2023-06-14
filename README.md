mlwhatif
================================

[![mlwhatif](https://img.shields.io/badge/❓-mlwhatif-green)](https://github.com/stefan-grafberger/mlwhatif)
[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://github.com/stefan-grafberger/mlwhatif/blob/master/LICENSE)
[![Build Status](https://github.com/stefan-grafberger/mlwhatif/actions/workflows/build.yml/badge.svg)](https://github.com/stefan-grafberger/mlwhatif/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/stefan-grafberger/mlwhatif/branch/main/graph/badge.svg?token=KTMNPBV1ZZ)](https://codecov.io/gh/stefan-grafberger/mlwhatif)

Data-Centric What-If Analysis for Native Machine Learning Pipelines.

This project uses the [mlinspect](https://github.com/stefan-grafberger/mlinspect) project as a foundation, mainly for its plan extraction from native ML pipelines.

## Run mlwhatif locally

Prerequisite: Python 3.9

1. Clone this repository (optionally, with [Git LFS](https://github.com/git-lfs/git-lfs), to also download the datasets for the scalability experiment)
2. Set up the environment

	`cd mlwhatif` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>

3. If you want to use the visualisation functions we provide, install graphviz which can not be installed via pip

    `Linux: ` `apt-get install graphviz` <br>
    `MAC OS: ` `brew install graphviz` <br>
	
4. Install pip dependencies 

    `SETUPTOOLS_USE_DISTUTILS=stdlib pip install -e ."[dev]"` <br>

5. To ensure everything works, you can run the tests (without graphviz, the visualisation test will fail)

    `python setup.py test` <br>

## How to use mlwhatif
mlwhatif makes it easy to analyze your pipeline and automatically run what-if analyses.
```python
from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis import DataCleaning, ErrorType

IPYNB_PATH = ...
cleanlearn = DataCleaning({'category': ErrorType.CAT_MISSING_VALUES,
                           'vine': ErrorType.CAT_MISSING_VALUES,
                           'star_rating': ErrorType.NUM_MISSING_VALUES,
                           'total_votes': ErrorType.OUTLIERS,
                           'review_id': ErrorType.DUPLICATES,
                           None: ErrorType.MISLABEL
                         })

analysis_result = PipelineAnalyzer \
    .on_pipeline_from_ipynb_file(IPYNB_PATH)\
    .add_what_if_analysis(cleanlearn) \
    .execute()

cleanlearn_report = analysis_result.analysis_to_result_reports[cleanlearn]
```

## Detailed Example
We prepared a [demo notebook](demo/feature_overview/feature_overview.ipynb) to showcase mlwhatif and its features.

## Notes
* For debugging in PyCharm, set the pytest flag `--no-cov` ([Link](https://stackoverflow.com/questions/34870962/how-to-debug-py-test-in-pycharm-when-coverage-is-enabled))
* If you want to see log output in PyCharm, you can also set the pytest flags `--log-cli-level=10 -s`. The `-s` is needed because otherwise pytest breaks the stdout capturing.

## Publications
* Stefan Grafberger, Shubha Guha, Paul Groth, Sebastian Schelter (2023). mlwhatif: What If You Could Stop Re-Implementing Your Machine Learning Pipeline Analyses Over and Over? VLDB (demo).
* [Stefan Grafberger, Paul Groth, Sebastian Schelter (2023). Automating and Optimizing Data-Centric What-If Analyses
on Native Machine Learning Pipelines. ACM SIGMOD.](https://stefan-grafberger.com/mlwhatif.pdf)
* [Stefan Grafberger, Paul Groth, Sebastian Schelter (2022). Towards Data-Centric What-If Analysis for Native Machine Learning Pipelines. Data Management for End-to-End Machine Learning workshop at ACM SIGMOD.](https://stefan-grafberger.com/mlwhatif-deem.pdf)

## License
This library is licensed under the Apache 2.0 License.
