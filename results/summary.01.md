# Trial Convergence Summary

RL convergence may have been caused by floating point accuracy.

| Avg Reward | Device | Pretraining Epochs | Equation Sample Count | Runtime Cache Size | RL Batch Size | RL Learning Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -3.0347 | **CUDA(GB10)** | 100 | 10,000 | 10,000 | 32| 1e-4 |
| -1.8391 | **CUDA(RTX3090)** | 100 | 10,000 | 10,000 | 32| 1e-4 |
| 0.0473  | **CUDA(RTX3090)** | 1,000 | 10,000 | 10,000 | 32 | 1e-4 |
| 3.8703  | **CPU** | 100 | 10,000 | 10,000 | 32| 1e-4 |
| 5.6525  | **CUDA(GB10)** | 100 | 30,000 | 100,000 | 32 | 1e-4 |
| 7.3284  | **CUDA(GB10)** | 100 | 10,000 | 10,000 | 32| 1e-4 |
| 8.3655  | **CUDA(GB10)** | 100 | 10,000 | 10,000 | 32 | 1e-4 |


