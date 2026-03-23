# Parallel-torchsort

## Overview

This work parallelizes Teddy Koker's implementation of [torchsort](https://github.com/teddykoker/torchsort) for the sequence length dimension. For that, we used a Divide-&-Conquer approach to parallelize the Pool Adjacent Violators (PAV) Algorithm. We demonstrated the parallelization and its correctness by conducting wall-clock time benchmarking, correctness unit tests, and classification training on [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) using top-K loss. 