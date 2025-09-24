# Collaborative Knowledge Editing for Large Language Model Services in Edge-Cloud Computing


## Table of Contents

- [Installation](#installation)
- [Running the Full Evaluation Suite](#running-the-full-evaluation-suite)
- [Generating Scaling Curves](#generating-scaling-curves)
- [How to Cite](#how-to-cite)

## Installation

We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`.


## Running the Full Evaluation Suite

[`coedit/evaluate.py`](coedit/evaluate.py) can be used to evaluate any method in [`baselines/`](baselines/).

For example:
```
python -m coedit.evaluate \
    --alg_name=MEMIT \
    --model_name=EleutherAI/gpt-j-6B \
    --hparams_fname=EleutherAI_gpt-j-6B.json \
    --ds_name=zsre \
    --generation_test_interval=10 
    --num_clients=5 \
    --num-time=10\
```
Results from each run are stored at `results/<method_name>/run_<run_id>` in a specific format:
```bash
results/
|__ MEMIT/
    |__ run_<run_id>/
        |__client_<client_id>/
            |__time_<time_id>/
                |__ params.json
                |__ case_0.json
                |__ case_1.json
                |__ ...
                |__ case_10000.json
```

To summarize the results, you can use [`coedit/summarize.py`](coedit/summarize.py):
```bash
python -m coedit.summarize --dir_name=MEMIT --runs=run_<run1>/client_<client0>/_time_<time0>.
```

Running `python -m coedit.evaluate -h` provides details about command-line flags.

## How to Cite

```bibtex
```
