"""Human-readable NAS-Bench-201 architecture summaries."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Sequence

from nas_benchmarks.constants import NUM_NB201_EDGES, NUM_NB201_OPS

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
NODE_NAMES = {1: "input", 2: "node_1", 3: "node_2", 4: "output"}
OP_NAMES = (
    "Identity",
    "Zero",
    "ReLUConvBN3x3",
    "ReLUConvBN1x1",
    "AvgPool1x1",
)
NB201_OP_NAMES = (
    "skip_connect",
    "none",
    "nor_conv_3x3",
    "nor_conv_1x1",
    "avg_pool_3x3",
)


def parse_architecture(raw_arch: str) -> tuple[int, ...]:
    """Parse a genotype string such as '(2, 3, 0, 1, 2, 3)'."""
    raw_arch = raw_arch.strip()
    try:
        parsed = ast.literal_eval(raw_arch)
    except (SyntaxError, ValueError):
        parsed = raw_arch

    if isinstance(parsed, int):
        values = (parsed,)
    elif isinstance(parsed, str):
        values = tuple(_parse_int(value) for value in parsed.split(",") if value.strip())
    elif isinstance(parsed, Sequence):
        values = tuple(_parse_int(value) for value in parsed)
    else:
        raise ValueError(f"Architecture must be a tuple/list of ints, got {type(parsed)!r}")

    validate_architecture(values)
    return values


def validate_architecture(arch: tuple[int, ...]) -> None:
    if len(arch) != NUM_NB201_EDGES:
        raise ValueError(f"NAS-Bench-201 genotype must have 6 ops, got {len(arch)}")

    invalid_ops = [op for op in arch if op < 0 or op >= NUM_NB201_OPS]
    if invalid_ops:
        raise ValueError(
            "NAS-Bench-201 op indices must be in range 0..4, "
            f"got {sorted(set(invalid_ops))}"
        )


def format_architecture_summary(
    arch: tuple[int, ...],
    input_shape: tuple[int, int, int] = (3, 32, 32),
    num_classes: int = 10,
) -> str:
    """Return torchsummary-style text for one NAS-Bench-201 genotype."""
    validate_architecture(arch)
    c_in, height, width = input_shape
    channels = (16, 32, 64)
    stem_params = _stem_param_count(c_in, channels[0])
    stage_1_params = _cell_param_count(arch, channels[0]) * 5
    stage_2_params = _cell_param_count(arch, channels[1]) * 5
    stage_3_params = _cell_param_count(arch, channels[2]) * 5
    reduction_1_params = _resnet_block_param_count(channels[0], channels[1])
    reduction_2_params = _resnet_block_param_count(channels[1], channels[2])
    classifier_params = _classifier_param_count(channels[2], num_classes)
    total_params = (
        stem_params
        + stage_1_params
        + stage_2_params
        + stage_3_params
        + reduction_1_params
        + reduction_2_params
        + classifier_params
    )
    shape = (channels[0], height, width)
    after_reduction_1 = (channels[1], _halve(height), _halve(width))
    after_reduction_2 = (
        channels[2],
        _halve(after_reduction_1[1]),
        _halve(after_reduction_1[2]),
    )

    lines = [
        "NAS-Bench-201 Architecture Summary",
        "==================================",
        f"Genotype: {arch}",
        f"NB201 string: {format_nb201_string(arch)}",
        f"Valid architecture: {'yes' if is_valid_nb201_arch(arch) else 'no'}",
        f"Input shape: [N, {c_in}, {height}, {width}]",
        "",
        "Operation Legend",
        "----------------",
        _format_table(
            ("Index", "NASLib op", "NAS-Bench-201 op", "Meaning"),
            [
                ("0", "Identity", "skip_connect", "pass input unchanged"),
                ("1", "Zero", "none", "remove connection"),
                ("2", "ReLUConvBN3x3", "nor_conv_3x3", "3x3 conv block"),
                ("3", "ReLUConvBN1x1", "nor_conv_1x1", "1x1 conv block"),
                ("4", "AvgPool1x1", "avg_pool_3x3", "3x3 average pool"),
            ],
        ),
        "",
        "Cell DAG",
        "--------",
        _format_table(
            ("#", "Edge", "From", "To", "Op", "NASLib op", "NB201 op"),
            [
                (
                    str(index),
                    f"{src}->{dst}",
                    NODE_NAMES[src],
                    NODE_NAMES[dst],
                    str(op),
                    OP_NAMES[op],
                    NB201_OP_NAMES[op],
                )
                for index, ((src, dst), op) in enumerate(zip(EDGE_LIST, arch))
            ],
        ),
        "",
        "Node Equations",
        "--------------",
        *format_node_equations(arch),
        "",
        "Macro Network",
        "-------------",
        _format_table(
            ("Layer (type)", "Output shape", "Param #", "Details"),
            [
                (
                    "Stem",
                    _shape_text(shape),
                    _format_count(stem_params),
                    f"Conv2d({c_in}->{channels[0]}, 3x3) + BatchNorm2d",
                ),
                (
                    "Cell x5",
                    _shape_text(shape),
                    _format_count(stage_1_params),
                    "stage_1, shared genotype, stride=1",
                ),
                (
                    "ResNetBasicblock",
                    _shape_text(after_reduction_1),
                    _format_count(reduction_1_params),
                    f"{channels[0]}->{channels[1]}, stride=2",
                ),
                (
                    "Cell x5",
                    _shape_text(after_reduction_1),
                    _format_count(stage_2_params),
                    "stage_2, shared genotype, stride=1",
                ),
                (
                    "ResNetBasicblock",
                    _shape_text(after_reduction_2),
                    _format_count(reduction_2_params),
                    f"{channels[1]}->{channels[2]}, stride=2",
                ),
                (
                    "Cell x5",
                    _shape_text(after_reduction_2),
                    _format_count(stage_3_params),
                    "stage_3, shared genotype, stride=1",
                ),
                (
                    "Classifier",
                    f"[N, {num_classes}]",
                    _format_count(classifier_params),
                    (
                        "BatchNorm2d -> ReLU -> AdaptiveAvgPool2d(1) "
                        f"-> Flatten -> Linear({channels[2]}->{num_classes})"
                    ),
                ),
            ],
        ),
        f"Total params: {_format_count(total_params)}",
    ]
    return "\n".join(lines)


def format_node_equations(arch: tuple[int, ...]) -> list[str]:
    edge_ops = {edge: OP_NAMES[op] for edge, op in zip(EDGE_LIST, arch)}
    equations = []
    for node in (2, 3, 4):
        terms = [
            f"{edge_ops[(src, node)]}({NODE_NAMES[src]})"
            for src in range(1, node)
        ]
        equations.append(f"{NODE_NAMES[node]} = " + " + ".join(terms))
    return equations


def is_valid_nb201_arch(arch: tuple[int, ...]) -> bool:
    return not (
        (arch[0] == arch[1] == arch[2] == 1)
        or (arch[2] == arch[4] == arch[5] == 1)
    )


def format_nb201_string(arch: tuple[int, ...]) -> str:
    edge_op_dict = {edge: NB201_OP_NAMES[op] for edge, op in zip(EDGE_LIST, arch)}
    op_edge_list = [
        f"{edge_op_dict[(src, dst)]}~{src - 1}"
        for src, dst in sorted(edge_op_dict, key=lambda edge: edge[1])
    ]
    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a NAS-Bench-201 genotype into a readable summary."
    )
    parser.add_argument(
        "architecture",
        help="Tuple/list/comma-separated op indices, e.g. '(2,3,0,1,2,3)'.",
    )
    parser.add_argument(
        "--input-shape",
        default="3,32,32",
        help="Input shape as C,H,W. Default: 3,32,32.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Classifier output classes. Default: 10.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional text file path. Defaults to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        arch = parse_architecture(args.architecture)
        input_shape = parse_input_shape(args.input_shape)
    except ValueError as error:
        parser.error(str(error))

    summary = format_architecture_summary(
        arch,
        input_shape=input_shape,
        num_classes=args.num_classes,
    )

    if args.output is None:
        print(summary)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary + "\n", encoding="utf-8")


def parse_input_shape(raw_shape: str) -> tuple[int, int, int]:
    values = tuple(_parse_int(value) for value in raw_shape.split(",") if value.strip())
    if len(values) != 3:
        raise ValueError(f"Input shape must be C,H,W, got {raw_shape!r}")
    if any(value <= 0 for value in values):
        raise ValueError(f"Input shape dimensions must be positive, got {raw_shape!r}")
    return values


def _parse_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected integer op index, got {value!r}") from error


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(len(str(row[index])) for row in (headers, *rows))
        for index in range(len(headers))
    ]
    header_line = "  ".join(
        str(header).ljust(width) for header, width in zip(headers, widths)
    )
    rule_line = "  ".join("-" * width for width in widths)
    row_lines = [
        "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths))
        for row in rows
    ]
    return "\n".join((header_line, rule_line, *row_lines))


def _shape_text(shape: tuple[int, int, int]) -> str:
    channels, height, width = shape
    return f"[N, {channels}, {height}, {width}]"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _cell_param_count(arch: tuple[int, ...], channels: int) -> int:
    return sum(_op_param_count(op, channels) for op in arch)


def _op_param_count(op: int, channels: int) -> int:
    if op == 2:
        return _relu_conv_bn_param_count(channels, channels, kernel_size=3)
    if op == 3:
        return _relu_conv_bn_param_count(channels, channels, kernel_size=1)
    return 0


def _stem_param_count(c_in: int, c_out: int) -> int:
    return c_in * c_out * 3 * 3 + _batch_norm_param_count(c_out)


def _resnet_block_param_count(c_in: int, c_out: int) -> int:
    return (
        _relu_conv_bn_param_count(c_in, c_out, kernel_size=3)
        + _relu_conv_bn_param_count(c_out, c_out, kernel_size=3)
        + c_in * c_out
    )


def _classifier_param_count(c_in: int, num_classes: int) -> int:
    return _batch_norm_param_count(c_in) + c_in * num_classes + num_classes


def _relu_conv_bn_param_count(c_in: int, c_out: int, kernel_size: int) -> int:
    return c_in * c_out * kernel_size * kernel_size + _batch_norm_param_count(c_out)


def _batch_norm_param_count(channels: int) -> int:
    return channels * 2


def _halve(value: int) -> int:
    return max(1, value // 2)


if __name__ == "__main__":
    main()
