import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_dir():
    for path in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (path / "MCPBRNN_lib_tools").is_dir() and (path / "20220527-MDUPLEX-LeafRiver").is_dir():
            return path
    raise FileNotFoundError(
        "Could not locate project root containing MCPBRNN_lib_tools "
        "and 20220527-MDUPLEX-LeafRiver."
    )


PROJECT_DIR = find_project_dir()
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS
from MCPBRNN_lib_tools.MCP_Zoo import MCPBRNN_PETconstraint_MassRelax_Regular_Relaxed

# IMPORTANT: scaling comes from the historical EVAL script.
C_MEAN = 475.0253683
C_STD = 89.86101728
SPLIT_NAMES = {-99999: "spinup", -1: "train", 0: "selection", 1: "testing"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MR3: Regular Relaxed mass relaxation.")
    p.add_argument("--time_lag", type=int, default=0)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def resolve_path(value, default):
    path = Path(value) if value else default
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def locate_checkpoint(value):
    if value:
        path = Path(value)
        path = path if path.is_absolute() else (SCRIPT_DIR / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    files = sorted(p for p in SCRIPT_DIR.glob("*.pt") if p.is_file())
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one .pt in {SCRIPT_DIR}, found {len(files)}.")
    return files[0]


def safe(v):
    return -99999 if np.isnan(v) else float(v)


def calc_metrics(sim_t, obs_t):
    sim, obs = sim_t.detach().cpu().numpy(), obs_t.detach().cpu().numpy()
    kge, corr, a, b, kgess = KGE(sim, obs)
    return {
        "NSE": float(NS(sim, obs)),
        "KGE": safe(kge),
        "KGE-A": float(a),
        "KGE-B": float(b),
        "Corr": safe(corr),
        "MSE": float(mean_squared_error(obs, sim)),
        "KGEss": safe(kgess),
    }


def build_time_axis(n):
    spin = pd.date_range("1948-10-01", "1949-09-30", freq="D")
    sim = pd.date_range("1948-10-01", "1988-09-30", freq="D")
    time, phase = [], []

    for i in range(1, 4):
        time.extend(spin.strftime("%Y-%m-%d"))
        phase.extend([f"spinup{i}"] * len(spin))

    time.extend(sim.strftime("%Y-%m-%d"))
    phase.extend(["simulation"] * len(sim))

    if len(time) != n:
        raise ValueError(f"Expected {len(time)} rows, found {n}.")
    return time, phase


def save_series(path, time, phase, tensor, name):
    values = tensor.detach().cpu().numpy().reshape(-1)
    pd.DataFrame({"time": time, name: values, "phase": phase}).to_csv(path, index=False)


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
        raise ValueError("Data and flag lengths differ.")

    x = torch.tensor(data[["P", "PET"]].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    y = torch.tensor(data[["Q"]].to_numpy(), dtype=torch.float32, device=device)
    f = torch.tensor(flags.to_numpy(), device=device)

    masks = {
        "train": f.eq(-1).unsqueeze(1),
        "selection": f.eq(0).unsqueeze(1),
        "testing": f.eq(1).unsqueeze(1),
        "spinup": f.eq(-99999).unsqueeze(1),
    }
    return data, flags, x, y, masks


class Model(nn.Module):
    def __init__(self, spin_len, train_len):
        super().__init__()
        self.MCPBRNNNode = MCPBRNN_PETconstraint_MassRelax_Regular_Relaxed(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            gate_dim_ucorr=3,
            spinLen=spin_len,
            traintimeLen=train_len,
            initial_forget_bias=0,
        )

    def forward(self, x, time_lag, y):
        return self.MCPBRNNNode(x, 0, time_lag, y, C_MEAN, C_STD)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = locate_checkpoint(args.checkpoint)
    data_dir = resolve_path(args.data_dir, PROJECT_DIR / "20220527-MDUPLEX-LeafRiver")
    output_dir = resolve_path(args.output_dir, SCRIPT_DIR / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    spin_len, train_len = 1095 - args.time_lag, 8400 - args.time_lag
    data, flags, x, y, masks = load_data(data_dir, device)

    model = Model(spin_len, train_len).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()

    with torch.no_grad():
        result = model(x, args.time_lag, y)

    q = result[0]
    rows = []

    for split in ("train", "selection", "testing", "spinup"):
        sim = torch.masked_select(q, masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)
        row = {"split": split, "n": int(sim.numel())}
        row.update(calc_metrics(sim, obs))
        rows.append(row)

    time, phase = build_time_axis(len(x))
    ts = pd.DataFrame({
        "time": time,
        "Qsim": q.detach().cpu().numpy().reshape(-1),
        "phase": phase,
        "P": data["P"].to_numpy(),
        "PET": data["PET"].to_numpy(),
        "Q": data["Q"].to_numpy(),
        "Flag": flags.to_numpy(),
    })
    ts["Split"] = ts["Flag"].map(SPLIT_NAMES)

    ts.to_csv(output_dir / "evaluation_timeseries.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "evaluation_metrics.csv", index=False)

    outputs = [
        ("Qsim.csv", result[0], "discharge"),
        ("storage.csv", result[1], "storage"),
        ("loss_unconstrained.csv", result[2], "loss_unconstrained"),
        ("loss_constrained.csv", result[3], "loss_constrained"),
        ("bypass.csv", result[4], "bypass"),
        ("gate_input.csv", result[5], "gate_input"),
        ("gate_output.csv", result[6], "gate_output"),
        ("gate_loss_unconstrained.csv", result[7], "gate_loss_unconstrained"),
        ("gate_loss_constrained.csv", result[8], "gate_loss_constrained"),
        ("gate_remember.csv", result[9], "gate_remember"),
        ("gate_mass_relaxation.csv", result[12], "mass_relaxation_gate"),
    ]

    for filename, tensor, name in outputs:
        save_series(output_dir / filename, time, phase, tensor, name)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nScaling: mean={C_MEAN}, std={C_STD}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
