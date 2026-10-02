import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS  # noqa: E402
from MCPBRNN_lib_tools.Loss_Function import KGELoss  # noqa: E402
from MCPBRNN_lib_tools.MCP_Zoo import (  # noqa: E402
    MCPBRNN_Generic_PETconstraint_MIoutput_Sigmoid,
)


PARAMETER_COLUMNS = [
    "MCPBRNNNode.weight_r_yom",
    "MCPBRNNNode.weight_r_ylm",
    "MCPBRNNNode.weight_r_yfm",
    "MCPBRNNNode.bias_b0_yom",
    "MCPBRNNNode.weight_b1_yom",
    "MCPBRNNNode.weight_b2_yom",
    "MCPBRNNNode.bias_b0_ylm",
    "MCPBRNNNode.weight_b2_ylm",
]

METRIC_NAMES = ["NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss"]
SPLITS = ("train", "selection", "testing", "spinup")

C_MEAN = 412.9139085
C_STD = 77.49943867


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Continue training the MI-Only output-gate model with a direct "
            "current/lagged-storage sigmoid output gate."
        )
    )
    parser.add_argument("--case_no", type=int, default=0)
    parser.add_argument("--epoch_no", type=int, default=3000)
    parser.add_argument("--time_lag", type=int, default=1)
    parser.add_argument("--seed_no", type=int, default=2925)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
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

    candidates = sorted(p for p in model_dir.glob("*.pt") if p.is_file())

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
        "mse": float(mean_squared_error(obs, sim)),
        "KGEss": safe_value(kgess),
    }


def calculate_lag_kge(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()
    n = obs.shape[0]

    scores = []
    for lag in (1, 2, 3):
        kge, *_ = KGE(sim[lag:n], obs[: n - lag])
        scores.append(safe_value(kge))

    return scores


def build_time_axis(n_rows):
    """Three repeated WY1949 spinups followed by WY1949-WY1988."""
    spinup_dates = pd.date_range("1948-10-01", "1949-09-30", freq="D")
    simulation_dates = pd.date_range("1948-10-01", "1988-09-30", freq="D")

    time = []
    phase = []

    for cycle in range(1, 4):
        time.extend(spinup_dates.strftime("%Y-%m-%d"))
        phase.extend([f"spinup{cycle}"] * len(spinup_dates))

    time.extend(simulation_dates.strftime("%Y-%m-%d"))
    phase.extend(["simulation"] * len(simulation_dates))

    if len(time) != n_rows:
        raise ValueError(
            f"Expected {len(time)} records for 3 x WY1949 spinup + "
            f"WY1949-WY1988, but found {n_rows}."
        )

    return time, phase


def save_timeseries(path, time, phase, tensor, column_name):
    values = tensor.detach().cpu().numpy().reshape(-1)

    pd.DataFrame(
        {
            "time": time,
            column_name: values,
            "phase": phase,
        }
    ).to_csv(path, index=False)


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
        raise ValueError("Forcing/flow data and skill flags have different lengths.")

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

    flag_t = torch.tensor(flags.to_numpy(), device=device)

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }

    return x, y, masks


class Model(nn.Module):
    """Thin wrapper retained so historical checkpoint keys remain compatible."""

    def __init__(self, spin_len, train_time_len):
        super().__init__()

        self.MCPBRNNNode = MCPBRNN_Generic_PETconstraint_MIoutput_Sigmoid(
            input_size=1,
            hidden_size=1,
            gate_dim_o=1,
            gate_dim_l=1,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )

    def forward(self, x, epoch, time_lag, y_obs):
        return self.MCPBRNNNode(
            x,
            epoch,
            time_lag,
            y_obs,
            C_MEAN,
            C_STD,
        )


def parameter_values(model):
    state = model.state_dict()

    return [
        state[name].detach().cpu().reshape(-1)[0].item()
        for name in PARAMETER_COLUMNS
    ]


def main():
    args = parse_args()

    if args.epoch_no < 1:
        raise ValueError("epoch_no must be at least 1.")

    np.random.seed(args.seed_no)
    torch.manual_seed(args.seed_no)

    device = resolve_device(args.device)
    time_lag = args.time_lag

    spin_len = 1095 - time_lag
    train_time_len = 8400 - time_lag

    learning_rate = 0.025
    learning_rates = {300: 0.0125, 600: 0.0125}

    model_dir = SCRIPT_DIR

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
        model_dir / "training_output",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, masks = load_data(data_dir, device)

    model = Model(
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

    loss_func = KGELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []
    best_epoch = None
    best_kge = -np.inf
    best_state = None

    for epoch in range(1, args.epoch_no + 1):
        if epoch in learning_rates:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch]

        model.train()
        optimizer.zero_grad()

        discharge = model(
            x,
            epoch,
            time_lag,
            y,
        )[0]

        sim_train = torch.masked_select(
            discharge,
            masks["train"],
        ).unsqueeze(1)

        obs_train = torch.masked_select(
            y,
            masks["train"],
        ).unsqueeze(1)

        loss = loss_func(
            sim_train,
            obs_train,
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            result = model(
                x,
                epoch,
                time_lag,
                y,
            )

            discharge = result[0]
            split_tensors = {}
            row = {"epoch": epoch}

            for split in SPLITS:
                sim = torch.masked_select(
                    discharge,
                    masks[split],
                ).unsqueeze(1)

                obs = torch.masked_select(
                    y,
                    masks[split],
                ).unsqueeze(1)

                split_tensors[split] = (sim, obs)
                metrics = calculate_metrics(sim, obs)

                suffix = "" if split == "train" else f"_{split}"

                for metric_name in METRIC_NAMES:
                    row[f"{metric_name}{suffix}"] = metrics[metric_name]

        for name, value in zip(
            PARAMETER_COLUMNS,
            parameter_values(model),
        ):
            row[name] = value

        for lag, score in zip(
            (1, 2, 3),
            calculate_lag_kge(*split_tensors["train"]),
        ):
            row[f"KGEtimelag_{lag}"] = score

        history.append(row)

        if row["KGE_selection"] > best_kge:
            best_kge = row["KGE_selection"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch}: "
            f"KGE_selection = {row['KGE_selection']:.6f}"
        )

    summary_columns = (
        ["epoch"]
        + PARAMETER_COLUMNS
        + METRIC_NAMES
        + [f"{name}_selection" for name in METRIC_NAMES]
        + [f"{name}_testing" for name in METRIC_NAMES]
        + [f"{name}_spinup" for name in METRIC_NAMES]
        + ["KGEtimelag_1", "KGEtimelag_2", "KGEtimelag_3"]
    )

    pd.DataFrame(history)[summary_columns].to_csv(
        output_dir / f"IC_caseno_{args.case_no}_summary.csv",
        index=False,
    )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    model.eval()

    with torch.no_grad():
        result = model(
            x,
            best_epoch,
            time_lag,
            y,
        )

    time, phase = build_time_axis(x.shape[0])

    output_series = {
        f"Outhidden_{args.case_no}_summary.csv": (
            result[0],
            "discharge",
        ),
        f"Outcell_{args.case_no}_summary.csv": (
            result[1],
            "storage",
        ),
        f"Outloss_{args.case_no}_summary.csv": (
            result[2],
            "loss_unconstrained",
        ),
        f"Outlossc_{args.case_no}_summary.csv": (
            result[3],
            "loss_constrained",
        ),
        f"Outbypass_{args.case_no}_summary.csv": (
            result[4],
            "bypass",
        ),
        f"Out_gatei_{args.case_no}_summary.csv": (
            result[5],
            "gate_input",
        ),
        f"Out_gateo_{args.case_no}_summary.csv": (
            result[6],
            "gate_output",
        ),
        f"Out_gatel_{args.case_no}_summary.csv": (
            result[7],
            "gate_loss_unconstrained",
        ),
        f"Out_gatelc_{args.case_no}_summary.csv": (
            result[8],
            "gate_loss_constrained",
        ),
        f"Out_gateremember_{args.case_no}_summary.csv": (
            result[9],
            "gate_remember",
        ),
    }

    for filename, (tensor, column_name) in output_series.items():
        save_timeseries(
            output_dir / filename,
            time,
            phase,
            tensor,
            column_name,
        )

    pd.DataFrame(
        [
            {
                "Best_Epoch": best_epoch,
                "KGE_selection": best_kge,
            }
        ]
    ).to_csv(
        output_dir / f"Best_Epoch_Caseno{args.case_no}_summary.csv",
        index=False,
    )

    final_checkpoint = (
        output_dir / f"best_model_epoch{best_epoch}.pt"
    )

    torch.save(
        best_state,
        final_checkpoint,
    )

    print(f"Initial checkpoint: {checkpoint}")
    print(f"Best epoch = {best_epoch}")
    print(f"Best KGE_selection = {best_kge:.6f}")
    print(f"Saved best model to: {final_checkpoint}")


if __name__ == "__main__":
    main()
