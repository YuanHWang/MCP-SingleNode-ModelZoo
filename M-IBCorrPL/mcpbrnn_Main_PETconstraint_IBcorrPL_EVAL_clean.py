import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS  # noqa: E402
from MCPBRNN_lib_tools.MCP_Zoo import (  # noqa: E402
    MCPBRNN_PETconstraint_IBcorrPL_Generic,
)


SPLIT_NAMES = {
    -99999: "spinup",
    -1: "train",
    0: "selection",
    1: "testing",
}

# Storage scaling inherited from the M5 PET-constrained reference model.
C_MEAN = 412.9139085
C_STD = 77.49943867


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pretrained piecewise-linear precipitation-bias-corrected "
            "single-node MCP (dimension 1-5). No training is performed."
        )
    )
    parser.add_argument(
        "--dim",
        "--gate_dim_ucorr",
        dest="dim",
        type=int,
        choices=range(1, 6),
        default=1,
        help="Piecewise-linear input-bias-correction dimension (1-5).",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--time_lag", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def resolve_path(value, default_path):
    path = Path(value) if value else default_path
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def locate_checkpoint(checkpoint_arg, model_dir):
    if checkpoint_arg:
        checkpoint = Path(checkpoint_arg)
        if not checkpoint.is_absolute():
            checkpoint = (model_dir / checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    candidates = sorted(
        p for p in model_dir.glob("*.pt")
        if p.is_file()
    )

    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) if candidates else "none"
        raise RuntimeError(
            f"Expected exactly one .pt file in {model_dir}, found {len(candidates)} "
            f"({names}). Use --checkpoint to specify the file explicitly."
        )

    return candidates[0]


def safe_value(value):
    return -99999 if np.isnan(value) else float(value)


def calculate_metrics(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()

    kge, corr, kge_a, kge_b, kgess = KGE(sim, obs)

    return {
        "NSE": float(NS(sim, obs)),
        "KGE": safe_value(kge),
        "KGE-A": float(kge_a),
        "KGE-B": float(kge_b),
        "Corr": safe_value(corr),
        "MSE": float(mean_squared_error(obs, sim)),
        "KGEss": safe_value(kgess),
    }


def build_time_axis(n_rows):
    """Three repeated WY1949 spinups followed by WY1949-WY1988 simulation."""
    spinup_dates = pd.date_range(
        "1948-10-01",
        "1949-09-30",
        freq="D",
    )
    simulation_dates = pd.date_range(
        "1948-10-01",
        "1988-09-30",
        freq="D",
    )

    time = []
    phase = []

    for cycle in range(1, 4):
        time.extend(
            spinup_dates.strftime("%Y-%m-%d")
        )
        phase.extend(
            [f"spinup{cycle}"] * len(spinup_dates)
        )

    time.extend(
        simulation_dates.strftime("%Y-%m-%d")
    )
    phase.extend(
        ["simulation"] * len(simulation_dates)
    )

    if len(time) != n_rows:
        raise ValueError(
            f"Expected {len(time)} records for 3 x WY1949 spinup + "
            f"WY1949-WY1988, but found {n_rows}."
        )

    return time, phase


def save_timeseries(
    path,
    time,
    phase,
    tensor,
    column_name,
):
    values = tensor.detach().cpu().numpy().reshape(-1)

    pd.DataFrame(
        {
            "time": time,
            column_name: values,
            "phase": phase,
        }
    ).to_csv(
        path,
        index=False,
    )


def load_data(data_dir, device):
    data = pd.read_csv(
        data_dir / "LeafRiverDaily_43YR.txt",
        header=None,
        sep=r"\s+",
        names=["P", "PET", "Q"],
    )

    flags = pd.read_csv(
        data_dir / "LeafRiverDaily_43YR_Flag.txt",
        header=None,
        sep=r"\s+",
        names=["Flag"],
    )["Flag"]

    if len(data) != len(flags):
        raise ValueError(
            "Forcing/flow data and skill flags have different lengths."
        )

    x = torch.tensor(
        data[["P", "PET"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    y = torch.tensor(
        data[["Q"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    flag_t = torch.tensor(
        flags.to_numpy(),
        device=device,
    )

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }

    return data, flags, x, y, masks


class Model(nn.Module):
    """Thin wrapper retained so historical checkpoint keys remain compatible."""

    def __init__(
        self,
        dim,
        spin_len,
        train_time_len,
    ):
        super().__init__()

        self.MCPBRNNNode = (
            MCPBRNN_PETconstraint_IBcorrPL_Generic(
                input_size=1,
                hidden_size=1,
                gate_dim=1,
                gate_dim_ucorr=dim,
                spinLen=spin_len,
                traintimeLen=train_time_len,
                initial_forget_bias=0,
            )
        )

    def forward(
        self,
        x,
        time_lag,
        y_obs,
    ):
        return self.MCPBRNNNode(
            x,
            0,
            time_lag,
            y_obs,
            C_MEAN,
            C_STD,
        )


def evaluate_splits(
    predictions,
    y,
    masks,
):
    rows = []

    for split in (
        "train",
        "selection",
        "testing",
        "spinup",
    ):
        sim = torch.masked_select(
            predictions,
            masks[split],
        ).unsqueeze(1)

        obs = torch.masked_select(
            y,
            masks[split],
        ).unsqueeze(1)

        row = {
            "split": split,
            "n": int(sim.numel()),
        }
        row.update(
            calculate_metrics(sim, obs)
        )
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    device = resolve_device(args.device)
    time_lag = args.time_lag

    model_dir = (
        SCRIPT_DIR
        / f"M{args.dim}-IBCorrPL"
    )

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Expected dimension folder: {model_dir}"
        )

    checkpoint = locate_checkpoint(
        args.checkpoint,
        model_dir,
    )

    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )

    output_dir = resolve_path(
        args.output_dir,
        model_dir / "evaluation",
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spin_len = 1095 - time_lag
    train_time_len = 8400 - time_lag

    data, flags, x, y, masks = load_data(
        data_dir,
        device,
    )

    model = Model(
        args.dim,
        spin_len,
        train_time_len,
    ).to(device)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
        ),
        strict=True,
    )

    model.eval()

    with torch.no_grad():
        result = model(
            x,
            time_lag,
            y,
        )

    predictions = result[0]

    metrics = evaluate_splits(
        predictions,
        y,
        masks,
    )

    time, phase = build_time_axis(
        x.shape[0]
    )

    timeseries = pd.DataFrame(
        {
            "time": time,
            "Qsim": predictions.detach()
            .cpu()
            .numpy()
            .reshape(-1),
            "phase": phase,
            "P": data["P"].to_numpy(),
            "PET": data["PET"].to_numpy(),
            "Q": data["Q"].to_numpy(),
            "Flag": flags.to_numpy(),
        }
    )

    timeseries["Split"] = (
        timeseries["Flag"]
        .map(SPLIT_NAMES)
    )

    timeseries.to_csv(
        output_dir / "evaluation_timeseries.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir / "evaluation_metrics.csv",
        index=False,
    )

    diagnostics = {
        "Qsim.csv": (
            result[0],
            "discharge",
        ),
        "storage.csv": (
            result[1],
            "storage",
        ),
        "loss_unconstrained.csv": (
            result[2],
            "loss_unconstrained",
        ),
        "loss_constrained.csv": (
            result[3],
            "loss_constrained",
        ),
        "bypass.csv": (
            result[4],
            "bypass",
        ),
        "gate_input.csv": (
            result[5],
            "gate_input",
        ),
        "gate_output.csv": (
            result[6],
            "gate_output",
        ),
        "gate_loss_unconstrained.csv": (
            result[7],
            "gate_loss_unconstrained",
        ),
        "gate_loss_constrained.csv": (
            result[8],
            "gate_loss_constrained",
        ),
        "gate_remember.csv": (
            result[9],
            "gate_remember",
        ),
        "input_bias_correction.csv": (
            result[10],
            "input_bias_correction",
        ),
    }

    for filename, (
        tensor,
        column_name,
    ) in diagnostics.items():
        save_timeseries(
            output_dir / filename,
            time,
            phase,
            tensor,
            column_name,
        )

    print(metrics.to_string(index=False))
    print(f"\nDimension:  {args.dim}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Outputs:    {output_dir}")


if __name__ == "__main__":
    main()
