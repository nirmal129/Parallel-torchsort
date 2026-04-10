# Parallel-torchsort

## Overview

This work parallelizes Teddy Koker's implementation of [torchsort](https://github.com/teddykoker/torchsort) for the sequence length dimension. For that, we used a Divide-&-Conquer approach to parallelize the Pool Adjacent Violators (PAV) Algorithm. We demonstrated the parallelization and its correctness by conducting wall-clock time benchmarking, correctness unit tests, and classification training on [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) using top-K loss. 

## Modifications and Results

The kernels implementing parallel PAV are in [isotonic_cuda.cu](https://github.com/nirmal129/torchsort/blob/main/torchsort/isotonic_cuda.cu#L304) and the results are in [extra/](https://github.com/nirmal129/torchsort/tree/main/extra). 

![Parallel Benchmark L2](https://github.com/nirmal129/torchsort/blob/main/torchsort/extra/benchmark_cuda_nnp_l2.png)

![Parallel Benchmark KL](https://github.com/nirmal129/torchsort/blob/main/torchsort/extra/benchmark_cuda_nnp_kl.png)