"""Tabular DARTS-style proxy optimizer for NAS-Bench-201."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nas_benchmarks.constants import NUM_NB201_EDGES, NUM_NB201_OPS
from nas_benchmarks.nb201 import (
    arch_to_tensor,
    is_valid_nb201_arch,
    query_nb201_arch,
    random_valid_arch,
)


class SurrogateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_NB201_EDGES * NUM_NB201_OPS, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TabularDARTSProxyOptimizer:
    def __init__(self, args: argparse.Namespace, dataset_api: dict[str, Any]):
        self.args = args
        self.dataset_api = dataset_api
        self.alpha = nn.Parameter(torch.zeros(NUM_NB201_EDGES, NUM_NB201_OPS))
        self.alpha_optimizer = torch.optim.Adam([self.alpha], lr=args.darts_lr)
        self.surrogate = SurrogateMLP()
        self.surrogate_optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=1e-3)
        self.x_train: list[torch.Tensor] = []
        self.y_train: list[float] = []
        self.best_arch: tuple[int, ...] | None = None
        self.best_metrics: dict[str, float] | None = None

    def step(self, epoch: int) -> dict[str, Any]:
        sampled_arch, entropy = self._sample_arch()
        sampled_metrics = query_nb201_arch(
            sampled_arch, self.args.dataset, self.dataset_api
        )
        reward = sampled_metrics["val_acc"] / 100.0
        self.x_train.append(arch_to_tensor(sampled_arch))
        self.y_train.append(reward)
        self._update_best(sampled_arch, sampled_metrics)

        loss_value = 0.0
        if len(self.x_train) >= 2:
            loss_value = self._fit_surrogate()
        if len(self.x_train) >= 2 and epoch + 1 >= self.args.darts_warmup:
            loss_value = self._update_alpha()

        return {
            "sampled_arch": sampled_arch,
            "best_arch": self.best_arch,
            "best_metrics": self.best_metrics,
            "reward": reward,
            "loss": loss_value,
            "entropy": entropy,
        }

    def _sample_arch(self) -> tuple[tuple[int, ...], float]:
        for _ in range(100):
            probs = torch.softmax(self.alpha / self.args.darts_temperature, dim=-1)
            sampled = []
            entropies = []
            for edge_probs in probs:
                distribution = torch.distributions.Categorical(probs=edge_probs)
                token = distribution.sample()
                sampled.append(int(token.item()))
                entropies.append(distribution.entropy())
            arch = tuple(sampled)
            if is_valid_nb201_arch(arch):
                return arch, float(torch.stack(entropies).sum().detach().item())
        return random_valid_arch(), 0.0

    def _fit_surrogate(self) -> float:
        x_train = torch.stack(self.x_train)
        y_train = torch.tensor(self.y_train, dtype=torch.float32)
        loss = torch.tensor(0.0)
        for _ in range(self.args.darts_surrogate_epochs):
            prediction = self.surrogate(x_train)
            loss = F.mse_loss(prediction, y_train)
            self.surrogate_optimizer.zero_grad()
            loss.backward()
            self.surrogate_optimizer.step()
        return float(loss.detach().item())

    def _update_alpha(self) -> float:
        probs = torch.softmax(self.alpha / self.args.darts_temperature, dim=-1)
        soft_arch = probs.flatten().unsqueeze(0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        prediction = self.surrogate(soft_arch).mean()
        loss = -prediction - self.args.rl_entropy_weight * entropy

        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        return float(loss.detach().item())

    def _update_best(self, arch: tuple[int, ...], metrics: dict[str, float]) -> None:
        if self.best_metrics is None or metrics["val_acc"] > self.best_metrics["val_acc"]:
            self.best_arch = arch
            self.best_metrics = metrics

    def final_metadata(self) -> dict[str, Any]:
        alpha_argmax_arch = tuple(int(x) for x in torch.argmax(self.alpha, dim=-1).tolist())
        if not is_valid_nb201_arch(alpha_argmax_arch):
            alpha_argmax_arch = (
                self.best_arch if self.best_arch is not None else random_valid_arch()
            )
        alpha_metrics = query_nb201_arch(
            alpha_argmax_arch, self.args.dataset, self.dataset_api
        )
        return {
            "alpha_argmax_arch": str(alpha_argmax_arch),
            "alpha_argmax_metrics": alpha_metrics,
        }

