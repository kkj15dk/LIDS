"""Run training and observation simulation for a fixed number of rounds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import main as train_script
import simulate_observation as observation_script
from timing import print_time


def _extract_hydra_override(args: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _parse_duplicate_index_arg(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("duplicate-index must not be empty")
    indices: list[int] = []
    for part in parts:
        try:
            index = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"duplicate-index must be an int, got {part!r}"
            ) from None
        if index < 0:
            raise argparse.ArgumentTypeError(
                f"duplicate-index must be >= 0, got {index}"
            )
        indices.append(index)
    return indices


def _read_project_path_from_config(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("project_path:"):
            value = line.split(":", 1)[1].strip()
            return value if value else None
    return None


def _build_duplicate_overrides(
    forwarded_args: list[str],
    base_project_path: str | None,
    duplicate_idx: int,
    duplicate_count: int,
) -> list[str]:
    duplicate_overrides = [
        arg
        for arg in forwarded_args
        if not arg.startswith("project_path=")
        and not arg.startswith("round=")
        and not arg.startswith("theta_path=")
        and not arg.startswith("simulation.theta_index=")
    ]

    if base_project_path is not None:
        duplicate_project_path = (
            base_project_path
            if duplicate_count == 1
            else os.path.join(base_project_path, f"duplicate_{duplicate_idx:03d}")
        )
        duplicate_overrides.insert(0, f"project_path={duplicate_project_path}")

    duplicate_overrides.insert(0, f"simulation.theta_index={duplicate_idx}")
    return duplicate_overrides


def _compose_config(config_path: Path, overrides: list[str]) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {config_path}, got {type(cfg)}")
    if not overrides:
        return cfg

    override_cfg = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(cfg, override_cfg)
    if not isinstance(merged, DictConfig):
        raise TypeError("Merged configuration is not a DictConfig")
    return merged


def _run_duplicate(
    *,
    duplicate_idx: int,
    rounds: int,
    repo_root: Path,
    forwarded_args: list[str],
    base_project_path: str | None,
    duplicate_count: int,
    stop_on_perfect_map: bool,
) -> None:
    duplicate_overrides = _build_duplicate_overrides(
        forwarded_args=forwarded_args,
        base_project_path=base_project_path,
        duplicate_idx=duplicate_idx,
        duplicate_count=duplicate_count,
    )
    config_path = repo_root / "conf" / "config.yaml"

    # reuse the oracle across runs, prevent rebuilding it.
    probe_cfg = _compose_config(config_path=config_path, overrides=["round=1", *duplicate_overrides])
    if str(probe_cfg.oracle.type) == "plm":
        with print_time(f"duplicate={duplicate_idx} build_plm_oracle"):
            cached_oracle = observation_script.build_plm_oracle(probe_cfg, device=str(probe_cfg.device))
    else:
        cached_oracle = None

    # 1. prior samples
    for round_idx in range(1, rounds + 1):
        # 2.a train
        train_overrides = [
            f"round={round_idx}",
            "theta_path=null",
            *duplicate_overrides,
        ]
        print(
            f"Running duplicate={duplicate_idx} round={round_idx} train with overrides: {train_overrides}",
            flush=True,
        )
        train_cfg = _compose_config(config_path=config_path, overrides=train_overrides)
        with print_time(f"duplicate={duplicate_idx} round={round_idx} train"):
            train_script._main(train_cfg)

        # 2.b observe
        observation_overrides = [
            f"round={round_idx}",
            *duplicate_overrides,
        ]
        print(
            f"Running duplicate={duplicate_idx} round={round_idx} observe with overrides: {observation_overrides}",
            flush=True,
        )
        observation_cfg = _compose_config(config_path=config_path, overrides=observation_overrides)
        with print_time(f"duplicate={duplicate_idx} round={round_idx} observe"):
            map_ratio = observation_script._main(observation_cfg, oracle=cached_oracle)
        if stop_on_perfect_map and map_ratio is not None and map_ratio >= 1.0:
            print(
                f"MAP_ratio reached {map_ratio:.4f} at duplicate={duplicate_idx} round={round_idx}. "
                "Skipping remaining rounds for this duplicate.",
                flush=True,
            )
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main.py and simulate_observation.py for N rounds.")
    parser.add_argument("--rounds", type=int, required=True, help="Number of rounds to run.")
    parser.add_argument("--duplicates", type=int, default=1, help="Number of duplicate experiment runs.")
    parser.add_argument(
        "--duplicate-index",
        action="append",
        type=_parse_duplicate_index_arg,
        default=None,
        help=(
            "Run only selected duplicate index(es). Repeat the flag or pass a comma-separated list."
        ),
    )
    parser.add_argument(
        "--project-path",
        dest="project_path",
        type=str,
        default=None,
        help="Alias of --project_path.",
    )
    parser.add_argument(
        "--project_path",
        type=str,
        default=None,
        help="Hydra project_path override shared by both scripts.",
    )
    parser.add_argument(
        "--continue-after-perfect-map",
        dest="stop_on_perfect_map",
        action="store_false",
        help="Keep running rounds even after MAP_ratio reaches 1.0.",
    )
    args, extra_hydra_args = parser.parse_known_args()

    if args.rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {args.rounds}")
    if args.duplicates < 1:
        raise ValueError(f"duplicates must be >= 1, got {args.duplicates}")
    duplicate_indices: list[int] | None = None
    if args.duplicate_index is not None:
        duplicate_indices = [idx for group in args.duplicate_index for idx in group]
        for idx in duplicate_indices:
            if idx >= args.duplicates:
                raise ValueError(
                    "duplicate-index entries must be < duplicates "
                    f"({args.duplicates}), got {idx}"
                )

    repo_root = Path(__file__).resolve().parent

    forwarded_args = list(extra_hydra_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    if args.project_path is not None:
        project_path_arg = f"project_path={args.project_path}"
        forwarded_args = [arg for arg in forwarded_args if not arg.startswith("project_path=")]
        forwarded_args.insert(0, project_path_arg)

    base_project_path = args.project_path or _extract_hydra_override(forwarded_args, "project_path")
    if base_project_path is None:
        base_project_path = _read_project_path_from_config(repo_root / "conf" / "config.yaml")
    if args.duplicates > 1 and base_project_path is None:
        raise ValueError("--duplicates requires project_path (override or conf/config.yaml) so duplicate runs do not overwrite each other.")

    forwarded_args = [
        arg
        for arg in forwarded_args
        if not arg.startswith("round=") and not arg.startswith("theta_path=")
    ]

    if duplicate_indices is not None:
        for duplicate_idx in duplicate_indices:
            _run_duplicate(
                duplicate_idx=duplicate_idx,
                rounds=args.rounds,
                repo_root=repo_root,
                forwarded_args=forwarded_args,
                base_project_path=base_project_path,
                duplicate_count=args.duplicates,
                stop_on_perfect_map=bool(args.stop_on_perfect_map),
            )
        return

    for duplicate_idx in range(args.duplicates):
        _run_duplicate(
            duplicate_idx=duplicate_idx,
            rounds=args.rounds,
            repo_root=repo_root,
            forwarded_args=forwarded_args,
            base_project_path=base_project_path,
            duplicate_count=args.duplicates,
            stop_on_perfect_map=bool(args.stop_on_perfect_map),
        )


if __name__ == "__main__":
    main()