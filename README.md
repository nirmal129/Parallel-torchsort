# Parallel-torchsort

## Overview

This work parallelizes Teddy Koker's implementation of [torchsort](https://github.com/teddykoker/torchsort) for the sequence length dimension. For that, we used a Divide-&-Conquer approach to parallelize the Pool Adjacent Violators (PAV) Algorithm. We demonstrated the parallelization and its correctness by conducting wall-clock time benchmarking, correctness unit tests, and classification training on [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) using top-K loss. 

## Modifications and Results

The kernels implementing parallel PAV are in [isotonic_cuda.cu](https://github.com/nirmal129/torchsort/blob/main/torchsort/isotonic_cuda.cu#L304) and the results are in [extra/](https://github.com/nirmal129/torchsort/tree/main/extra). 

![Parallel Benchmark L2](https://github.com/nirmal129/torchsort/raw/main/extra/benchmark_cuda_nnp_l2.png)

![Parallel Benchmark KL](https://github.com/nirmal129/torchsort/raw/main/extra/benchmark_cuda_nnp_kl.png)

## Execution Instructions

First, load the required modules on TACC. I used Vista but any GPU allocation should work. 

```
module load gcc/14.2.0 cmake/4.1.1 openmpi/5.0.5 ucx/1.20.0 ucc/1.7.0 \
            cuda/12.8 nccl/12.4 nvpl/26.1 python3_mpi/3.11.8 xalt/3.1
```

Then git clone [torchsort through this link](https://github.com/nirmal129/torchsort.git), install `torch>=1.7.1`, and run the below setup commands inside torchsort directory. 

```
export CUDA_HOME=$TACC_CUDA_DIR
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Let the active Python report its own header location.
# This overrides the hardcoded /usr/include/python3.9 that nvcc was picking up.
PY_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
export CPLUS_INCLUDE_PATH=$PY_INCLUDE:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$PY_INCLUDE:$C_INCLUDE_PATH

USE_NINJA=0 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="9.0" python setup.py build_ext --inplace
```

Finally, run the below commands from the parent directory of torchsort to run the unit tests and benchmarking, respectively, of the parallel implementation. 

```
TORCHSORT_PATH="$PWD/torchsort"
export PYTHONPATH=$TORCHSORT_PATH:$PYTHONPATH
cd torchsort
pytest tests/test_ops_nnp.py -v > ../test.log 2>&1
```
```
TORCHSORT_PATH="$PWD/torchsort"
export PYTHONPATH=$TORCHSORT_PATH:$PYTHONPATH
cd torchsort
python extra/benchmark_nnp.py --regularization l2
python extra/benchmark_nnp.py --regularization kl
```
