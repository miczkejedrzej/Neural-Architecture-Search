"""Local REINFORCE controller for tabular NAS-Bench-201 experiments."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn

from nas_benchmarks.constants import NUM_NB201_EDGES, NUM_NB201_OPS
from nas_benchmarks.nb201 import is_valid_nb201_arch, query_nb201_arch, random_valid_arch


class RLController(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(NUM_NB201_OPS + 1, hidden_size)
        self.cell = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, NUM_NB201_OPS)
        self.hidden_size = hidden_size

    def sample(self) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        token = torch.tensor([NUM_NB201_OPS], device=device)
        hidden = torch.zeros(1, self.hidden_size, device=device)
        cell = torch.zeros(1, self.hidden_size, device=device)
        log_probs = []
        entropies = []
        tokens = []

        for _ in range(NUM_NB201_EDGES):
            hidden, cell = self.cell(self.embedding(token), (hidden, cell))
            distribution = torch.distributions.Categorical(
                logits=self.output(hidden).squeeze(0)
            )
            token = distribution.sample().view(1)
            tokens.append(int(token.item()))
            log_probs.append(distribution.log_prob(token.squeeze(0)))
            entropies.append(distribution.entropy())

        return tuple(tokens), torch.stack(log_probs).sum(), torch.stack(entropies).sum()


class RLControllerOptimizer:
    def __init__(self, args: argparse.Namespace, dataset_api: dict[str, Any]):
        self.args = args
        self.dataset_api = dataset_api
        self.controller = RLController(args.rl_hidden_size)
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=args.rl_lr)
        self.baseline: float | None = None
        self.best_arch: tuple[int, ...] | None = None
        self.best_metrics: dict[str, float] | None = None

    def step(self) -> dict[str, Any]:
        for _ in range(100):
            sampled_arch, log_prob, entropy = self.controller.sample()
            if is_valid_nb201_arch(sampled_arch):
                break
        else:
            sampled_arch = random_valid_arch()
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)

        sampled_metrics = query_nb201_arch(
            sampled_arch, self.args.dataset, self.dataset_api
        )
        reward = sampled_metrics["val_acc"] / 100.0
        baseline = reward if self.baseline is None else self.baseline
        advantage = reward - baseline
        loss = -(advantage * log_prob) - self.args.rl_entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.baseline is None:
            self.baseline = reward
        else:
            momentum = self.args.rl_baseline_momentum
            self.baseline = momentum * self.baseline + (1.0 - momentum) * reward

        self._update_best(sampled_arch, sampled_metrics)
        return {
            "sampled_arch": sampled_arch,
            "best_arch": self.best_arch,
            "best_metrics": self.best_metrics,
            "reward": reward,
            "loss": float(loss.detach().item()),
            "entropy": float(entropy.detach().item()),
        }

    def _update_best(self, arch: tuple[int, ...], metrics: dict[str, float]) -> None:
        if self.best_metrics is None or metrics["val_acc"] > self.best_metrics["val_acc"]:
            self.best_arch = arch
            self.best_metrics = metrics

    def final_metadata(self) -> dict[str, Any]:
        return {}

