"""Range-only trajectory generator for matched and mismatched motion cases.

This script follows the experimental setup
- 2D constant-velocity state with process noise from the Wiener velocity model.
- Four fixed anchors with range-only measurements in the standard form.
- Two motion regimes: matched (omega=0) and mismatch (omega=pi/(2*L*dt) in turn windows).

Maneuvers (mismatch case) are injected as deterministic velocity rotations
inside a set of turn windows. By default, the generator uses a fixed turn
schedule to preserve backward compatibility with earlier experiments.
Optionally, each trajectory can sample its own random turn windows and random
turn directions (±omega).

Running the script writes datasets to:
- data/raw/trajectory       (matched)
- data/raw/trajectory_wpi   (mismatch)

Each dataset is stored as a single NPZ file named:
`trajectory_{w}_N{num}_T{steps}_qc{qc}_sigmar{sigma}.npz` (w in {w0, wpi}).
If you change the maneuver schedule (e.g., L/K or randomization), the filename
appends a suffix like `_L25_K6_randWin_randDir` to avoid overwriting older runs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


TURN_STARTS: Tuple[int, ...] = (50, 200, 350)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class GeneratorConfig:
    """Static configuration for dataset generation."""

    dt: float = 0.01
    steps: int = 500
    trajectories: int = 20
    qc: float = 0.5
    sigma_r: float = 0.05
    seed: int = 0
    tbptt_L: int = 50
    tbptt_stride: int = 50
    turn_length: int = 50  # left-closed, right-open window length
    turn_count: int = field(default_factory=lambda: len(TURN_STARTS))
    random_turn_windows: bool = False
    random_turn_directions: bool = False
    anchors: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-1.5, 1.0, -1.0, 1.5],
                [0.5, 1.0, -1.0, -0.5],
            ],
            dtype=float,
        )
    )
    # anchors[:, i] = (s_x^i, s_y^i)

    def fixed_turn_windows(self) -> np.ndarray:
        """Build the deterministic turn schedule used when random sampling is disabled."""
        steps = int(self.steps)
        L = int(self.turn_length)
        K = int(self.turn_count)
        if K <= 0:
            return np.zeros((0, 2), dtype=int)
        if L <= 0:
            raise ValueError(f"turn_length must be positive, got {L}")

        max_start = max(0, steps - L)
        if K == len(TURN_STARTS):
            starts = np.asarray(TURN_STARTS, dtype=int)
            starts = np.clip(starts, 0, max_start)
        else:
            starts = np.linspace(0, max_start, K, dtype=int)
        ends = starts + L
        windows = np.stack([starts, ends], axis=-1).astype(int)
        windows[:, 1] = np.clip(windows[:, 1], 0, steps)
        # Ensure end > start (non-empty).
        windows[:, 1] = np.maximum(windows[:, 1], windows[:, 0] + 1)
        return windows


@dataclass(frozen=True)
class CaseSpec:
    key: str
    base_omega: float
    output_subdir: str

    @property
    def mismatch(self) -> bool:
        return self.base_omega != 0.0


def build_case_specs(turn_rate: float) -> Dict[str, CaseSpec]:
    """Define matched vs mismatch cases; mismatch uses turn_rate computed from L, dt."""
    return {
        "matched": CaseSpec(key="w0", base_omega=0.0, output_subdir="trajectory"),
        "mismatch": CaseSpec(key="wpi", base_omega=turn_rate, output_subdir="trajectory_wpi"),
    }


def build_transition(dt: float) -> np.ndarray:
    """Constant-velocity transition matrix."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def build_process_noise(qc: float, dt: float) -> np.ndarray:
    """Discrete-time process noise covariance from the Wiener velocity model."""
    Q = np.array(
        [
            [qc * dt**3 / 3.0, 0.0, qc * dt**2 / 2.0, 0.0],
            [0.0, qc * dt**3 / 3.0, 0.0, qc * dt**2 / 2.0],
            [qc * dt**2 / 2.0, 0.0, qc * dt, 0.0],
            [0.0, qc * dt**2 / 2.0, 0.0, qc * dt],
        ],
        dtype=float,
    )
    return 0.5 * (Q + Q.T)


def compute_tbptt_blocks(steps: int, L: int, stride: int) -> np.ndarray:
    """Compute TBPTT block start/end indices (inclusive)."""
    blocks = []
    t = 0
    while t + L <= steps:
        blocks.append((t, t + L - 1))
        t += stride
    return np.array(blocks, dtype=int)


def _fmt_float(val: float) -> str:
    """Format float for filenames with trimmed trailing zeros."""
    return np.format_float_positional(val, trim="k")


def sample_turn_windows(
    rng: np.random.Generator,
    steps: int,
    turn_length: int,
    turn_count: int,
    *,
    max_tries: int = 512,
) -> np.ndarray:
    """Sample non-overlapping turn windows [start, end) for one trajectory."""
    steps = int(steps)
    L = int(turn_length)
    K = int(turn_count)

    if K <= 0:
        return np.zeros((0, 2), dtype=int)
    if L <= 0:
        raise ValueError(f"turn_length must be positive, got {L}")
    if steps < L:
        raise ValueError(f"steps must be >= turn_length, got steps={steps}, L={L}")

    max_start = steps - L
    population = max_start + 1
    if population <= 0:
        return np.zeros((0, 2), dtype=int)

    for _ in range(max_tries):
        replace = population < K
        starts = rng.choice(population, size=K, replace=replace).astype(int)
        starts.sort()
        windows = np.stack([starts, starts + L], axis=-1).astype(int)
        if np.all(windows[1:, 0] >= windows[:-1, 1]):
            return windows

    # Fallback: deterministic evenly-spaced schedule (always non-overlapping).
    starts = np.linspace(0, max_start, K, dtype=int)
    windows = np.stack([starts, starts + L], axis=-1).astype(int)
    windows[:, 1] = np.clip(windows[:, 1], 0, steps)
    windows[:, 1] = np.maximum(windows[:, 1], windows[:, 0] + 1)
    return windows


def build_omega_schedule(base_omega: float, steps: int, windows: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Expand sparse turn windows into a per-step angular rate schedule."""
    omega = np.zeros(int(steps), dtype=float)
    if base_omega == 0.0 or windows.size == 0:
        return omega

    if windows.ndim != 2 or windows.shape[1] != 2:
        raise ValueError(f"windows must have shape (K, 2), got {windows.shape}")
    if signs.shape[0] != windows.shape[0]:
        raise ValueError(f"signs must have shape (K,), got {signs.shape} for K={windows.shape[0]}")

    for (start, end), sign in zip(windows.astype(int), signs.astype(int)):
        if end <= start:
            continue
        omega[start:end] = float(sign) * float(base_omega)
    return omega


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    """Draw initial [x, y, vx, vy] from the specified uniforms."""
    px0 = rng.uniform(-0.5, 0.5)
    py0 = rng.uniform(-0.5, 0.5)
    vx0 = rng.uniform(0.5, 1.5)
    vy0 = rng.uniform(-0.5, 0.5)
    return np.array([px0, py0, vx0, vy0], dtype=float)


def simulate_trajectory(
    F: np.ndarray,
    Q: np.ndarray,
    cfg: GeneratorConfig,
    base_omega: float,
    rng: np.random.Generator,
    turn_windows: np.ndarray,
    turn_signs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one trajectory and its range measurements."""
    x = sample_initial_state(rng)
    states = np.zeros((cfg.steps, 4), dtype=float)
    states[0] = x

    omega_schedule = build_omega_schedule(base_omega, cfg.steps, turn_windows, turn_signs)
    for k in range(cfg.steps - 1):
        omega = float(omega_schedule[k])
        if omega != 0.0:
            c, s = np.cos(omega * cfg.dt), np.sin(omega * cfg.dt)
            vx, vy = x[2], x[3]
            x[2] = c * vx - s * vy
            x[3] = s * vx + c * vy
        w_k = rng.multivariate_normal(np.zeros(4, dtype=float), Q)
        x = F @ x + w_k
        states[k + 1] = x

    pos = states[:, :2]
    dx = pos[:, [0]] - cfg.anchors[0, :]
    dy = pos[:, [1]] - cfg.anchors[1, :]
    ranges = np.sqrt(dx * dx + dy * dy)
    noise = rng.normal(scale=cfg.sigma_r, size=ranges.shape)
    measurements = ranges + noise
    return states, measurements


def generate_dataset_for_case(spec: CaseSpec, cfg: GeneratorConfig) -> Dict[str, np.ndarray]:
    """Generate all trajectories for the specified motion case."""
    rng = np.random.default_rng(cfg.seed)
    F = build_transition(cfg.dt)
    Q = build_process_noise(cfg.qc, cfg.dt)
    fixed_windows = cfg.fixed_turn_windows()
    K = fixed_windows.shape[0]

    X = np.zeros((cfg.trajectories, cfg.steps, 4), dtype=float)
    Y = np.zeros((cfg.trajectories, cfg.steps, cfg.anchors.shape[1]), dtype=float)
    turn_windows_traj = np.zeros((cfg.trajectories, K, 2), dtype=int)
    turn_signs_traj = np.ones((cfg.trajectories, K), dtype=int)

    for n in range(cfg.trajectories):
        windows_n = fixed_windows
        signs_n = np.ones(K, dtype=int)
        if spec.mismatch:
            if cfg.random_turn_windows:
                windows_n = sample_turn_windows(rng, cfg.steps, cfg.turn_length, cfg.turn_count)
            if cfg.random_turn_directions and K > 0:
                signs_n = rng.choice(np.array([-1, 1], dtype=int), size=K)
        turn_windows_traj[n] = windows_n
        turn_signs_traj[n] = signs_n

        states, meas = simulate_trajectory(F, Q, cfg, spec.base_omega, rng, windows_n, signs_n)
        X[n] = states
        Y[n] = meas

    tbptt_single = compute_tbptt_blocks(cfg.steps, cfg.tbptt_L, cfg.tbptt_stride)
    tbptt_blocks = np.repeat(tbptt_single[None, :, :], cfg.trajectories, axis=0)

    train_ids = np.arange(0, 15, dtype=int)
    val_ids = np.arange(15, 18, dtype=int)
    test_ids = np.arange(18, 20, dtype=int)

    return {
        "case": np.array(spec.key),
        "mismatch": np.array(spec.mismatch),
        "omega_base": np.array(spec.base_omega, dtype=float),
        "turn_length": np.array(cfg.turn_length, dtype=int),
        "dt": np.array(cfg.dt, dtype=float),
        "steps": np.array(cfg.steps, dtype=int),
        "trajectories": np.array(cfg.trajectories, dtype=int),
        "qc": np.array(cfg.qc, dtype=float),
        "sigma_r": np.array(cfg.sigma_r, dtype=float),
        "anchors": cfg.anchors,
        "turn_count": np.array(cfg.turn_count, dtype=int),
        "random_turn_windows": np.array(cfg.random_turn_windows, dtype=bool),
        "random_turn_directions": np.array(cfg.random_turn_directions, dtype=bool),
        "turn_windows": fixed_windows,
        "turn_windows_traj": turn_windows_traj,
        "turn_signs_traj": turn_signs_traj,
        "F": F,
        "Q": Q,
        "X": X,
        "Y": Y,
        "tbptt_blocks": tbptt_blocks,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


def save_dataset(data: Dict[str, np.ndarray], root: Path, spec: CaseSpec) -> Path:
    """Save dataset to the appropriate case directory using the requested naming scheme."""
    out_dir = root / spec.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    num = int(data["trajectories"])
    steps = int(data["steps"])
    qc = _fmt_float(float(data["qc"]))
    sigma_r = _fmt_float(float(data["sigma_r"]))
    turn_length = int(data.get("turn_length", 50))
    turn_count = int(data.get("turn_count", len(TURN_STARTS)))
    rand_win = bool(data.get("random_turn_windows", False))
    rand_dir = bool(data.get("random_turn_directions", False))
    w_tag = spec.key

    suffix_parts = []
    if turn_length != 50:
        suffix_parts.append(f"L{turn_length}")
    if turn_count != len(TURN_STARTS):
        suffix_parts.append(f"K{turn_count}")
    if rand_win:
        suffix_parts.append("randWin")
    if rand_dir:
        suffix_parts.append("randDir")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

    filename = f"trajectory_{w_tag}_N{num}_T{steps}_qc{qc}_sigmar{sigma_r}{suffix}.npz"
    out_path = out_dir / filename
    np.savez(out_path, **data)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate range-only datasets for matched/mismatched cases.")
    parser.add_argument(
        "--case",
        choices=["matched", "mismatch", "all"],
        default="all",
        help="Which motion regime to generate. 'all' produces both matched and mismatch.",
    )
    parser.add_argument("--trajectories", type=int, default=20, help="Number of trajectories per case.")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps per trajectory.")
    parser.add_argument("--dt", type=float, default=0.01, help="Sampling period.")
    parser.add_argument("--qc", type=float, default=0.5, help="Continuous-time acceleration PSD.")
    parser.add_argument("--sigma-r", type=float, default=0.05, dest="sigma_r", help="Range noise std.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--turn-length", type=int, default=50, help="Turn window length (steps).")
    parser.add_argument(
        "--turn-count",
        type=int,
        default=len(TURN_STARTS),
        help="Number of turn windows in the mismatch case.",
    )
    parser.add_argument(
        "--random-turn-windows",
        action="store_true",
        help="Sample per-trajectory random non-overlapping turn windows (mismatch case only).",
    )
    parser.add_argument(
        "--random-turn-directions",
        action="store_true",
        help="Sample per-trajectory random turn directions (±omega) per window (mismatch case only).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Root output directory (defaults to data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GeneratorConfig(
        dt=args.dt,
        steps=args.steps,
        trajectories=args.trajectories,
        qc=args.qc,
        sigma_r=args.sigma_r,
        seed=args.seed,
        turn_length=args.turn_length,
        turn_count=args.turn_count,
        random_turn_windows=bool(args.random_turn_windows),
        random_turn_directions=bool(args.random_turn_directions),
    )

    # For mismatch: choose omega so that velocity rotates 90 deg over one window (Exercise 5.5 style).
    omega_turn = float(np.pi / (2.0 * cfg.turn_length * cfg.dt))
    case_specs = build_case_specs(omega_turn)
    case_keys = list(case_specs) if args.case == "all" else [args.case]

    for key in case_keys:
        spec = case_specs[key]
        data = generate_dataset_for_case(spec, cfg)
        out_path = save_dataset(data, args.out_root, spec)
        print(f"Saved {spec.key} dataset to {out_path}")


if __name__ == "__main__":
    main()
