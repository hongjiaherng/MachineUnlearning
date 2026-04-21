# Machine Unlearning

#### (Released on March 27, 2025)

## Introduction
This repository implements twelve machine unlearning algorithms in PyTorch under a common Hydra-driven interface. Three are reference points — `baseline` (no-op), `retrain` (oracle retrain from scratch on the retain set), and `fine_tune` (fine-tune on the retain set only) — and nine are published methods:

| Algorithm       | Paper                                                                                                                                               | Original Github Repository                            |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| gradient_ascent | [Unrolling SGD: Understanding Factors Influencing Machine Unlearning](https://arxiv.org/abs/2109.13398)                                             | https://github.com/cleverhans-lab/unrolling-sgd       |
| bad_teacher     | [Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher](https://arxiv.org/abs/2205.08096)                    | https://github.com/vikram2000b/bad-teaching-unlearning |
| scrub           | [Towards Unbounded Machine Unlearning](https://arxiv.org/abs/2302.09880)                                                                            | https://github.com/meghdadk/SCRUB/tree/main           |
| amnesiac        | [Amnesiac Machine Learning](https://arxiv.org/abs/2010.10981)                                                                                       | https://github.com/lmgraves/AmnesiacML                |
| boundary        | [Boundary Unlearning: Rapid Forgetting of Deep Networks via Shifting the Decision Boundary](https://ieeexplore.ieee.org/abstract/document/10203289) | https://github.com/TY-LEE-KR/Boundary-Unlearning-Code |
| ntk             | [Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible from Input-Output Observations](https://arxiv.org/abs/2003.02960)    | https://github.com/AdityaGolatkar/SelectiveForgetting |
| fisher          | [Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks](https://arxiv.org/abs/1911.04933)                                     | https://github.com/AdityaGolatkar/SelectiveForgetting |
| unsir           | [Fast Yet Effective Machine Unlearning](https://arxiv.org/abs/2111.08947)                                                                           | https://github.com/vikram2000b/Fast-Machine-Unlearning|
| ssd             | [Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening](https://arxiv.org/abs/2308.07707)                                 | https://github.com/if-loops/selective-synaptic-dampening|

Sincere appreciation to the authors of these popular machine unlearning algorithms for open-sourcing their code, greatly contributing to the success of this repository.

All strategies target **class unlearning** in the current harness — `split_unlearn_dataset` partitions the train set into *retain* (not of `unlearn_class`) and *forget* (the target class).
## Getting started

### Preparation

Uses [uv](https://docs.astral.sh/uv/). Requires Python 3.12.

```bash
uv sync --extra cpu      # CPU-only PyTorch
uv sync --extra cu130    # CUDA 13.0 PyTorch
```

### How to run

Both entrypoints are driven by [Hydra](https://hydra.cc). Compose a run by selecting options from
the `dataset`, `model`, `optimizer`, and `strategy` config groups, or override any field with `key=value`.

**1. Model Training**

```bash
uv run mu-train dataset=cifar10 model=resnet18
uv run mu-train dataset=mnist model=simplecnn epochs=1 batch_size=64
```

Checkpoints are written to `./checkpoint/{Model}/{scenario}/{Dataset}/`.

**2. Unlearning**

```bash
uv run mu-unlearn \
  dataset=cifar10 model=resnet18 \
  strategy=fine_tune \
  unlearn_class=0 \
  model_path=./checkpoint/ResNet18/class/CIFAR10/<checkpoint>.pt
```

The `retrain` strategy does not require `model_path`. Run `uv run mu-train --help` or
`uv run mu-unlearn --help` to see the full list of composable config groups.

Strategy-specific hyperparameters live under `strategy.params.*` and can be overridden on the command line:

```bash
uv run mu-unlearn dataset=mnist model=mlp strategy=gradient_ascent unlearn_class=0 \
  strategy.params.lr=1e-4 strategy.epochs=5 \
  model_path=./checkpoint/MLP/class/MNIST/<checkpoint>.pt
```

Hydra writes each run under `outputs/{train,unlearn}/<YYYYMMDD-HHMMSS>_<descriptor>/` with the resolved
config and logs, so runs are chronologically sortable by default.

### Evaluation

After each unlearning run, three metrics are reported:

- **Retain accuracy** — test accuracy restricted to non-target classes (higher is better).
- **Unlearn accuracy** — test accuracy on the forget class (lower is better).
- **MIA** — entropy-based membership-inference attack via logistic regression; lower attack success indicates better forgetting.

### Tests

```bash
uv run pytest
```

Runs the synthetic integration tests that exercise every strategy end-to-end on tiny tensors.

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the author by sending an email to
`winkent.ong at um.edu.my`.

# License and Copyright

The project is open source under BSD-3 license (see the `LICENSE` file).

©2024 Universiti Malaya.
