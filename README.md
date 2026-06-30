# Distributed GPU Training Cluster (WSL + Tailscale + PyTorch)

Google Colab gives you a single free GPU with session timeouts, usage limits, and no control over the underlying hardware. This project takes a different approach: instead of renting compute, it pools the idle GPUs/CPUs you already have across multiple personal machines into one distributed training cluster. There are no session limits, no queueing for availability, and no cost beyond the electricity bill — just your own laptops, networked together with Tailscale, doing real multi-node distributed deep learning.
 
This is a lightweight distributed systems project that demonstrates **multi-node distributed deep learning** using multiple Windows laptops connected over **Tailscale**. Each machine runs **WSL (Ubuntu)** with a Conda environment and participates in distributed training using **PyTorch Distributed Data Parallel (DDP)**.
 
The goal of this project is to pool the computational resources of multiple machines and train a neural network collaboratively over a network.
 

The goal of this project is to pool the computational resources of multiple machines and train a neural network collaboratively over a network.

---

## Features

- Multi-node distributed training
- Works across multiple Windows laptops
- WSL2 compatible
- Tailscale networking
- PyTorch Distributed Data Parallel (DDP)
- Automatic MNIST dataset download
- Simple neural network for testing
- Easy to scale from 2 → N nodes

---

## Project Structure

```text
distributed-training-cluster/
│
├── train_cluster.py
├── requirements.txt
├── README.md
└── data/
```

---

## Requirements

Each machine should have:

- Windows 10/11 or WSL 
- any Linux OS with good Conda support 
- NVIDIA GPU (recommended)
- Miniforge / Conda
- Python 3.10+
- Tailscale installed
- Internet connection (Minimum 100 mbps to avoid bottleneck)

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

## Step 2 — Create the Conda Environment

```bash
conda create -n gpu_cluster python=3.10 -y
conda activate gpu_cluster
```

## Step 3 — Install Dependencies

```bash
pip install torch torchvision
```

or

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
torchvision
```

## Step 4 — Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```

Verify that every machine reports the same PyTorch version.

## Step 5 — Install and Configure Tailscale

1. Install Tailscale on every machine.
2. Sign into the same Tailnet.
3. Verify connectivity.

Example:

```bash
tailscale status
```

You should see something similar to:

```text
100.x.x.x
100.x.x.x
100.x.x.x
100.x.x.x
```

Test communication:

```bash
ping <tailscale-ip>
```

## Step 6 — Choose the Master Node

Select one machine to act as the coordinator.

Example:

```text
MASTER NODE
IP: 100.xxx.xxx.xxx
```

Every other node connects to this machine.

## Step 7 — Configure Environment Variables

Every machine must use the same:

- `MASTER_ADDR`
- `MASTER_PORT`
- `WORLD_SIZE`

Only `RANK` changes.

**Master**

```bash
export MASTER_ADDR=100.xxx.xxx.xxx
export MASTER_PORT=43001

export WORLD_SIZE=2
export RANK=0

export GLOO_SOCKET_IFNAME=tailscale0
```

**Worker**

```bash
export MASTER_ADDR=100.xxx.xxx.xxx
export MASTER_PORT=43001

export WORLD_SIZE=2
export RANK=1

export GLOO_SOCKET_IFNAME=tailscale0
```

## Step 8 — Start the Cluster

Always launch the master first.

**Master**

```bash
python train_cluster.py
```

**Worker**

```bash
python train_cluster.py
```

For four nodes:

```text
Master      → RANK=0
Worker 1    → RANK=1
Worker 2    → RANK=2
Worker 3    → RANK=3
```

Set:

```bash
export WORLD_SIZE=4
```

on every machine.

---

## Dataset

The project automatically downloads the MNIST dataset during the first run. No manual download is required.

Dataset size: **~12 MB**

---

## Expected Output

**Master:**

```text
[CONNECTED] rank 0/2

Rank 0 starting training

Rank 0 finished epoch 0

MASTER: epoch completed
```

**Worker:**

```text
[CONNECTED] rank 1/2

Rank 1 starting training

Rank 1 finished epoch 0
```

If all nodes print `CONNECTED`, the process group has been successfully created.

---

## Scaling

Start with:

```bash
export WORLD_SIZE=2
```

Once successful, increase to:

```bash
export WORLD_SIZE=4
```

and assign:

```text
RANK=0
RANK=1
RANK=2
RANK=3
```

---

## Common Issues

### Address already in use

```text
EADDRINUSE
```

Choose another port:

```bash
export MASTER_PORT=43002
```

or check if another process is listening:

```bash
ss -lntp
```

### Workers Wait Forever

Usually caused by:

- Incorrect `WORLD_SIZE`
- Incorrect `RANK`
- Master not running
- Incorrect `MASTER_ADDR`

Verify all values match.

### Cannot Connect

Verify:

```bash
ping MASTER_ADDR
```

Check Tailscale:

```bash
tailscale status
```

### Wrong Network Interface

Use:

```bash
export GLOO_SOCKET_IFNAME=tailscale0
```

---

## Current Limitations

Current MVP supports:

- Static number of nodes
- Manual startup
- CPU backend (gloo)
- Manual environment configuration

Not yet implemented:

- GPU backend (nccl)
- Automatic node discovery
- Checkpoint recovery
- Elastic worker management
- Automatic SSH launch
- Performance dashboard

---

## Future Work

- GPU training using NCCL
- Checkpointing and resume
- Automatic cluster launcher
- Dynamic node addition/removal
- Fault tolerance
- Performance benchmarking
- Web dashboard
- Containerized deployment (Docker/Kubernetes)

---

## Project Architecture

```text
                 +----------------------+
                 |     Master Node      |
                 |      Rank = 0        |
                 | Process Coordinator  |
                 +----------+-----------+
                            |
        ---------------------------------------------
        |                    |                      |
        |                    |                      |
+---------------+    +---------------+    +---------------+
| Worker Rank 1 |    | Worker Rank 2 |    | Worker Rank 3 |
| Data Shard 1  |    | Data Shard 2  |    | Data Shard 3  |
+---------------+    +---------------+    +---------------+
        \                |                 /
         \               |                /
          \___________ Gradient Sync _____/
```

---

## Learning Outcomes

This project demonstrates:

- Distributed process coordination
- Synchronous distributed training
- Dataset sharding
- Gradient synchronization
- Multi-node communication
- Network debugging
- Distributed systems deployment using commodity hardware
- Practical experience with PyTorch Distributed Data Parallel

---

## Acknowledgements

Built as a distributed systems course project to explore practical distributed machine learning concepts.
