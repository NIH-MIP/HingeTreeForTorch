# Introduction
HingeTreeForTorch is a C++ PyTorch extension of RandomHingeForest.

# Tested Environments
HingeTreeForTorch has been built and tested in the following environments
- Arch Linux (circa Linux 6.16.8)
  - PyTorch 2.10.0 Nightly
  - Python 3.13.7
  - GCC 15.2.1
  - CUDA 13.0
- Ubuntu 20.04
  - PyTorch 1.12.1+cu113
  - Python 3.8.0
  - GCC 9.4.0
  - CUDA 11.3
- Rocky Linux 8.7
  - PyTorch 2.0.1+cu117
  - Python 3.9.15
  - GCC 9.2.0
  - CUDA 11.7
- OS X 10.15.7 (old information)
  - PyTorch 1.7.1
  - Python 3.8.2
  - XCode 12.4 (Clang 12.0.0)
- Windows 10
  - PyTorch 2.0.1+cpu
  - Python 3.11.4
  - Visual Studio 2022 Community Edition

# Compiling HingeTree
The HingeTree PyTorch extension can be compiled using the `build` python package.
```shell
python -m build --no-isolation
cd dist
pip install hingetree_cpp-<version>-<os>_<arch>.whl
```

**NOTE**: You need `--no-isolation` since PyTorch configurations can vary by GPU availability and CUDA version. Hence, HingeTree must be built against your environment.

**NOTE**: Make sure you delete the 'build' folder.

**NOTE**: Ensure that you are using Python 3.6 or later.

## Old Way
If you have trouble with the `build` module, this should still work (for now).
```shell
python setup.py bdist_wheel build
cd dist
pip install hingetree_cpp-<version>-<os>_<arch>.whl
```

## Mac OS X
If you experience any compiler errors, try the following
```shell
export ARCHFLAGS="-arch x86_64"
```
Then try again. Some recent version OS X sets this environment variable to simultaneously build for arm64 and x86_64!

## Windows 10
I have tested building HingeTreeForTorch with Visual Studio 2017 Community Edition and CUDA 10.2 against both PyTorch 1.7.1 and PyTorch 1.8.0 Nightly. I could not build HingeTreeForTorch in PyTorch 1.7.1 owing to uninitialized constexpr variables in PyTorch headers (nvcc error). If you experience this error, install PyTorch 1.8.0 Nightly and try again.

# Testing HingeTree
Once compiled and installed, you should be able to run the speed test
```py
import torch
from HingeTree import *
device = "cuda:0" if torch.cuda.is_available() else "cpu"
x = torch.rand([100, 1000]).to(device)
timings = HingeTree.speedtest(x)
```

# Basic Usage
There are two variants of tree-based learning machines, the `RandomHingeForest` and `RandomHingeFern`. They differ in their decision structure and parameter count, but otherwise behave identically

```py
from RandomHingeForest import RandomHingeForest, RandomHingeFern

forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth)
fern = RandomHingeFern(in_channels=numFeatures, out_channels=numTrees, depth=fernDepth)

# Process a batch of size 32
x = torch.rand([32, numFeatures])
y = forest(x)
```
By default, the `RandomHingeForest` and `RandomHingeFern` take in \[batchSize,numFeatures,d0,d1,...\] tensors and produce \[batchSize,numTrees,d0,d1...\] tensors. You can change the shape of predictions per leaf with `extra_outputs=outputShape`. For example
```py
forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth, extra_outputs=10) # Predict 10 values per tree
```
or
```py
forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth, extra_outputs=[8,8]) # Predict 8x8 values per tree
```
In which case the `RandomHingeForest` and `RandomHingeFern` take in \[batchSize,numFeatures,d0,d1,...\] tensors and produce \[batchSize,numTrees,d0,d1...,outputShape\] tensors.

# Reproducibility
Deterministic computation is now controlled by [PyTorch](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms). Forward passes are always deterministic on both the CPU and GPU. Backward passes are always deterministic on the CPU. The deterministic GPU backward pass can be very slow!

Some undocumented interfaces lack deterministic algorithms. These interfaces will raise `RuntimeError` exceptions when deterministic algorithms are not available. Both `RandomHingeForest` and `RandomHingeFern` support deterministic computation.

# Initialization
`RandomHingeForest` and `RandomHingeFern` support two types of initialization: random, sequential.

## `random`
Random initialization picks tree/fern decision thresholds on Uniform(-3,3), weights as Gaussian(0,1) and splitting feature indices on U\[0,numFeatures-1\]. This is the default initialization behavior.

```py
forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth, init_type="random")
```

## `sequential`
Sequential initialization picks tree/fern decision thresholds on Uniform(-3,3), weights as Gaussian(0,1). The spitting features are instead assigned sequentially in breadth-first-traversal fashion as:
```
vertexFeatureIndex = vertexIndex mod numFeatures
```
For example, you can use this to give each vertex a unique linear combination feature for decisions.
```py
import torch.nn as nn

numTrees=100
treeDepth=7
numFeatures=numTrees*(2**treeDepth - 1) # Give each vertex a unique linear combination feature

fc = nn.Linear(in_features=inputFeatures, out_features=numFeatures, bias=False)
bn = nn.BatchNorm1d(in_features=numFeatures, affine=False)
forest = RandomHingeForest(in_channels=numFeatures, out_channels=numTrees, depth=treeDepth, init_type="sequential")

# Evaluate a batch
x = forest(bn(fc(x)))
```

# Experiment Scripts
Set the `PYTHONPATH` environment variable to reflect the git root folder (the one with `HingeTree.py`, `RandomHingeForest.py`). Then you can run the experiments, for example, in this fashion:

```shell
cd experiments
python run_iris.py
```

And if you have a GPU
```shell
cd experiments
python run_iris.py --device cuda:0
```

## PYTHONPATH on Windows
```
cd C:\Path\To\HingeTreeForTorch
set pythonpath=%pythonpath%;%CD%
```

## PYTHONPATH Unix-like Systems
```shell
cd /path/to/HingeTreeForTorch
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

## Dependencies for Experiment Scripts
You will need `requests`, `scikit-learn`, `opencv-python`, `xgboost` and `lightgbm` to run these experiments.
```shell
pip install requests scikit-learn opencv-python xgboost lightgbm
```

All data sets are acquired automatically (usually downloaded from UCI machine learning repository).

## Errata 
If you see an error like this:

>RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

Just set/export this variable as follows
- Windows: `set CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Unix-like: `export CUBLAS_WORKSPACE_CONFIG=:4096:8`

And then try again.


