# RL-Based Neural Architecture Search — Complete Notes

---

## 1. Core Idea

Architecture design is framed as a **sequential decision process**: a controller agent emits a sequence of discrete tokens that fully specify a network architecture. That architecture is trained and evaluated, and the resulting validation accuracy becomes the **reward signal** to update the controller.

```
Controller → sample architecture a ~ π(a|θ)
           → train child network → get reward R (val accuracy)
           → update θ to increase E[R]
```

---

## 2. Controller Design

The controller is almost always an **RNN (LSTM)** that autoregressively produces tokens. Each timestep predicts one architectural decision:

```
t=1: number of filters in layer 1
t=2: filter height
t=3: filter width
t=4: stride
t=5: activation function
... (repeat per layer)
```

---

## 3. Training Algorithms

### 3.1 REINFORCE (Williams, 1992)

Since reward R is non-differentiable w.r.t. controller parameters θ:

$$\nabla_\theta J(\theta) = \frac{1}{m} \sum_{k=1}^{m} (R_k - b) \cdot \nabla_\theta \log \pi_\theta(a_k)$$

- **b** = exponential moving average of past rewards (variance reduction)
- **m** = mini-batch size of sampled architectures
- Controller updated with Adam at low LR (~3.5×10⁻⁴)

**Core problem:** reward sparsity + high variance gradient estimates.

### 3.2 PPO (MnasNet and later)

Clips large policy updates via the probability ratio:

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t = \pi_\theta(a_t) / \pi_{\theta_{old}}(a_t)$.

Benefits over REINFORCE:

- More stable updates
- Better sample efficiency
- Prevents policy collapse

---

## 4. Key Papers & Evolution

| Year | Paper | Key Contribution | Cost |
|------|-------|-----------------|------|
| 2017 | NAS (Zoph & Le) | LSTM + REINFORCE, first RL-NAS | ~22,400 GPU-days |
| 2018 | NASNet | Cell-based search, transferable cells | ~2,000 GPU-days |
| 2018 | ENAS | Weight sharing, shared supernet | ~0.45 GPU-days |
| 2019 | MnasNet | PPO + multi-objective compound reward, on-device latency | ~3,800 GPU-days |

---

## 5. Weight Sharing

### ENAS (naive)

All child architectures share a single supernet DAG. A sampled architecture is a subgraph — weights are the corresponding slice of the supernet.

**Two alternating phases:**

```
Phase 1 — Train shared weights w:
  Sample arch a ~ π_θ
  Forward/backward through child(a) using shared w
  Update w via SGD

Phase 2 — Train controller θ:
  Sample a ~ π_θ, evaluate on val set (no w update)
  Compute REINFORCE gradient, update θ via Adam
```

---

## 6. Reward Engineering

### Scalar (base)

Raw validation accuracy. High variance, no hardware awareness.

### Compound Multi-Objective (MnasNet)

$$R(m) = \text{ACC}(m) \times \left[\frac{\text{LAT}(m)}{T}\right]^w$$

- T = target latency on real device
- w = trade-off weight (typically −0.07)
- Latency measured on-device, not FLOPs

## 7. Controller Architecture Evolution

| Generation | Architecture | Limitation |
|-----------|--------------|-----------|
| 1st | Unidirectional LSTM | Sequential dependency — token t has no direct path to token t-20 |
| 2nd   | Bidirectional LSTM (EAS) | Better, but still limited context |
| 3rd | Transformer encoder | Global attention over all decisions simultaneously; parallelizable |

---