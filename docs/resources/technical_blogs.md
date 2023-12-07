# 大厂技术博客

## [PyTorch - Introducing PyTorch Fully Sharded Data Parallel (FSDP) API - CUDA Refresher (2022)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

> With PyTorch 1.11 we’re adding native support for Fully Sharded Data Parallel (FSDP), currently available as a prototype feature. Its implementation heavily borrows from FairScale’s version while bringing more streamlined APIs and additional performance improvements.

## [NVIDIA - CUDA Refresher (2020)](https://developer.nvidia.com/blog/tag/cuda-refresher/)

四篇CUDA科普博客

> Notes:

> 1. NVIDIA graphics processing units (GPUs) were originally designed for running games and graphics workloads that were highly parallel in nature.

> 2. GPGPU: general purpose computing on GPUs that were originally designed to accelerate only specific workloads like gaming and graphics. 

> 3. **CUDA is a parallel computing platform and programming model for general computing on graphical processing units (GPUs).** With CUDA, you can speed up applications by harnessing the power of GPUs. 

> 4. NVIDIA invented the CUDA programming model and addressed two challenges: simplify parallel programming to make it easy to program, and develop application software that transparently scales its parallelism to leverage the increasing number of processor cores with GPUs.

> 5. CUDA’s success was dependent on tools, libraries, applications, and partners available for CUDA ecosystem.

> 6. CUDA kernel is a function that gets executed on GPU. A group of threads is called a CUDA block. CUDA blocks are grouped into a grid. A kernel is executed as a grid of blocks of threads. Each CUDA block is executed by one streaming multiprocessor (SM) and cannot be migrated to other SMs in GPU (except during preemption, debugging, or CUDA dynamic parallelism). One SM can run several concurrent CUDA blocks depending on the resources needed by CUDA blocks.

> 7. The NVIDIA CUDA compiler does a good job in optimizing memory resources but **an expert CUDA developer can choose to use this memory hierarchy efficiently to optimize the CUDA programs as needed**.
