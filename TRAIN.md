# Reference Script for Training

## Training

To train models:

```bash
python train.py (--arg1 ... other args ... [data])
```

## Distributed data parallel training with TORCHRUN (ELASTIC LAUNCH)

### Single node training

To train models on a single node:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS train.py (--arg1 ... other args ... [data])
```

Or, you can use stacked single-node multi-worker:

```bash
torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:FREEPORT --nnodes=1 --nproc_per_node=$NUM_TRAINERS train.py (--arg1 ... other args ... [data])
```

### Multi node training

_Distributed training will be available via Slurm and submitit_

### Troubleshooting

To avoid `OMP_NUM_THREADS` warning:

```bash
export OMP_NUM_THREADS=$NUM_THREADS
```
