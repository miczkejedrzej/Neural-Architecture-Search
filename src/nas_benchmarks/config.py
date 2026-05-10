"""NASLib optimizer configuration helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from fvcore.common.config import CfgNode


def build_config(args: argparse.Namespace, optimizer: str) -> CfgNode:
    config = CfgNode(new_allowed=True)
    config.seed = args.seed
    config.optimizer = optimizer
    config.search_space = "nasbench201"
    config.dataset = args.dataset
    config.out_dir = args.out_dir
    config.save = str(Path(args.out_dir) / optimizer / f"seed_{args.seed}")
    config.save_arch_weights = False
    config.plot_arch_weights = False

    config.search = CfgNode(new_allowed=True)
    config.search.seed = args.seed
    config.search.epochs = args.max_queries
    config.search.checkpoint_freq = args.max_queries + 1
    config.search.sample_size = args.sample_size
    config.search.population_size = args.population_size

    config.search.predictor_type = args.predictor
    config.search.num_init = args.num_init
    config.search.k = args.bananas_k
    config.search.num_ensemble = args.num_ensemble
    config.search.acq_fn_type = "its"
    config.search.acq_fn_optimization = "mutation"
    config.search.encoding_type = "adjacency_one_hot"
    config.search.num_arches_to_mutate = args.num_arches_to_mutate
    config.search.max_mutations = args.max_mutations
    config.search.num_candidates = args.num_candidates
    config.search.debug_predictor = False
    config.search.zc = False
    config.search.use_zc_api = False
    config.search.zc_only = False
    config.search.zc_names = []
    config.search.load_labeled = False

    config.evaluation = CfgNode(new_allowed=True)
    return config
