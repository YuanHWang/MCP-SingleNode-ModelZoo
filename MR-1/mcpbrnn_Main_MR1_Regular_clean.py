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


def find_project_dir():
    for path in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (path / "MCPBRNN_lib_tools").is_dir() and (path / "20220527-MDUPLEX-LeafRiver").is_dir():
            return path
    raise FileNotFoundError("Could not locate project root containing MCPBRNN_lib_tools and 20220527-MDUPLEX-LeafRiver.")


PROJECT_DIR = find_project_dir()
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS
from MCPBRNN_lib_tools.Loss_Function import KGELoss
from MCPBRNN_lib_tools.MCP_Zoo import MCPBRNN_PETconstraint_MassRelax_Regular

# IMPORTANT: use the scaling values from the historical EVAL script.
C_MEAN = 543.3166693
C_STD = 79.61569861

PARAMETERS = [
    "MCPBRNNNode.weight_r_yom",
    "MCPBRNNNode.weight_r_ylm",
    "MCPBRNNNode.weight_r_yfm",
    "MCPBRNNNode.weight_r_yvm",
    "MCPBRNNNode.bias_b0_yom",
    "MCPBRNNNode.weight_b1_yom",
    "MCPBRNNNode.bias_b0_ylm",
    "MCPBRNNNode.weight_b2_ylm",
    "MCPBRNNNode.weight_s_yvm",
    "MCPBRNNNode.bias_b0_yrm",
]
METRICS = ["NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss"]
SPLITS = ("train", "selection", "testing", "spinup")


def parse_args():
    p = argparse.ArgumentParser(description="Continue training MR1: Regular mass relaxation.")
    p.add_argument("--case_no", type=int, default=0)
    p.add_argument("--epoch_no", type=int, default=3000)
    p.add_argument("--time_lag", type=int, default=0)
    p.add_argument("--seed_no", type=int, default=2925)
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
        "NSE": float(NS(sim, obs)), "KGE": safe(kge), "KGE-A": float(a), "KGE-B": float(b),
        "Corr": safe(corr), "mse": float(mean_squared_error(obs, sim)), "KGEss": safe(kgess),
    }


def lag_kge(sim_t, obs_t):
    sim, obs = sim_t.detach().cpu().numpy(), obs_t.detach().cpu().numpy()
    n = len(obs)
    return [safe(KGE(sim[k:n], obs[:n-k])[0]) for k in (1, 2, 3)]


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
    data = pd.read_csv(data_dir / "LeafRiverDaily_43YR.txt", header=None, sep=r"\s+", names=["P", "PET", "Q"])
    flags = pd.read_csv(data_dir / "LeafRiverDaily_43YR_Flag.txt", header=None, sep=r"\s+", names=["Flag"])["Flag"]
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
    return x, y, masks


class Model(nn.Module):
    def __init__(self, spin_len, train_len):
        super().__init__()
        self.MCPBRNNNode = MCPBRNN_PETconstraint_MassRelax_Regular(
            input_size=1, hidden_size=1, gate_dim=1, gate_dim_ucorr=3,
            spinLen=spin_len, traintimeLen=train_len, initial_forget_bias=0,
        )

    def forward(self, x, epoch, time_lag, y):
        return self.MCPBRNNNode(x, epoch, time_lag, y, C_MEAN, C_STD)


def parameter_values(model):
    s = model.state_dict()
    return [s[name].detach().cpu().reshape(-1)[0].item() for name in PARAMETERS]


def main():
    args = parse_args()
    if args.epoch_no < 1:
        raise ValueError("epoch_no must be >= 1.")

    np.random.seed(args.seed_no)
    torch.manual_seed(args.seed_no)
    device = resolve_device(args.device)

    checkpoint = locate_checkpoint(args.checkpoint)
    data_dir = resolve_path(args.data_dir, PROJECT_DIR / "20220527-MDUPLEX-LeafRiver")
    output_dir = resolve_path(args.output_dir, SCRIPT_DIR / "training_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    spin_len, train_len = 1095 - args.time_lag, 8400 - args.time_lag
    x, y, masks = load_data(data_dir, device)

    model = Model(spin_len, train_len).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)

    loss_func = KGELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    learning_rates = {300: 0.0125, 600: 0.0125}

    history, best_state, best_epoch, best_kge = [], None, None, -np.inf

    for epoch in range(1, args.epoch_no + 1):
        if epoch in learning_rates:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch]

        model.train()
        optimizer.zero_grad()

        q = model(x, epoch, args.time_lag, y)[0]
        sim = torch.masked_select(q, masks["train"]).unsqueeze(1)
        obs = torch.masked_select(y, masks["train"]).unsqueeze(1)

        loss = loss_func(sim, obs)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            result = model(x, epoch, args.time_lag, y)
            q = result[0]
            row, split_tensors = {"epoch": epoch}, {}

            for split in SPLITS:
                sim = torch.masked_select(q, masks[split]).unsqueeze(1)
                obs = torch.masked_select(y, masks[split]).unsqueeze(1)
                split_tensors[split] = (sim, obs)
                suffix = "" if split == "train" else f"_{split}"

                for name, value in calc_metrics(sim, obs).items():
                    row[f"{name}{suffix}"] = value

        for name, value in zip(PARAMETERS, parameter_values(model)):
            row[name] = value

        for lag, score in zip((1, 2, 3), lag_kge(*split_tensors["train"])):
            row[f"KGEtimelag_{lag}"] = score

        history.append(row)

        if row["KGE_selection"] > best_kge:
            best_kge, best_epoch = row["KGE_selection"], epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"epoch={epoch}, KGE_selection={row['KGE_selection']:.6f}")

    cols = (
        ["epoch"] + PARAMETERS + METRICS
        + [f"{m}_selection" for m in METRICS]
        + [f"{m}_testing" for m in METRICS]
        + [f"{m}_spinup" for m in METRICS]
        + ["KGEtimelag_1", "KGEtimelag_2", "KGEtimelag_3"]
    )
    pd.DataFrame(history)[cols].to_csv(output_dir / f"IC_caseno_{args.case_no}_summary.csv", index=False)

    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        result = model(x, best_epoch, args.time_lag, y)

    time, phase = build_time_axis(len(x))
    outputs = [
        ("Outhidden", result[0], "discharge"),
        ("Outcell", result[1], "storage"),
        ("Outloss", result[2], "loss_unconstrained"),
        ("Outlossc", result[3], "loss_constrained"),
        ("Outbypass", result[4], "bypass"),
        ("Out_gatei", result[5], "gate_input"),
        ("Out_gateo", result[6], "gate_output"),
        ("Out_gatel", result[7], "gate_loss_unconstrained"),
        ("Out_gatelc", result[8], "gate_loss_constrained"),
        ("Out_gateremember", result[9], "gate_remember"),
        ("Out_gatemr", result[12], "mass_relaxation_gate"),
        ("Out_fluxmr", result[13], "mass_relaxation_flux"),
    ]

    for prefix, tensor, name in outputs:
        save_series(output_dir / f"{prefix}_{args.case_no}_summary.csv", time, phase, tensor, name)

    pd.DataFrame([{"Best_Epoch": best_epoch, "KGE_selection": best_kge}]).to_csv(
        output_dir / f"Best_Epoch_Caseno{args.case_no}_summary.csv", index=False
    )

    final = output_dir / f"best_model_epoch{best_epoch}.pt"
    torch.save(best_state, final)

    print(f"Scaling: mean={C_MEAN}, std={C_STD}")
    print(f"Initial checkpoint: {checkpoint}")
    print(f"Best epoch: {best_epoch}; KGE_selection={best_kge:.6f}")
    print(f"Saved: {final}")


if __name__ == "__main__":
    main()
