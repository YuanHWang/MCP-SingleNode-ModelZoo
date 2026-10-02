# MCP SingleNode Model Zoo

This repository contains single-node **Mass-Conserving Perceptron (MCP)** model variants based on Wang & Gupta (2024), together with checkpoints used for training continuation, fine-tuning, and evaluation.

> Wang, Y.-H. and Gupta, H.V. (2024). *A mass-conserving-perceptron for machine-learning-based modeling of geoscientific systems*. Water Resources Research, 60(4), e2023WR036461.

## 1. Notation

The model notation used throughout the repository is:

| Symbol | Meaning |
|---|---|
| $O$ | Output gate |
| $L$ | Loss gate |
| $R$ | Remember gate |
| $\kappa$ | Constant gate |
| $\sigma$ | Sigmoid-based time-variable gate |
| $\mathrm{con}$ | PET-constrained loss gate |
| $B_{Lx}$ | Piecewise-linear precipitation bias-correction gate with dimension $x$ |
| $B_{Qx}$ | Piecewise-quadratic precipitation bias-correction gate with dimension $x$ |
| $A_x$ | Single hidden ANN layer with $x$ nodes |
| $+$ | Multi-information (MI): the gate uses additional state information |
| MR | Mass-relaxation mechanism |
---

## 2. Base single-node MCP models

The basic family contains constant (`κ`) and sigmoid-variable (`σ`) output/loss gates. For variable gates, the output gate depends on the current storage state and the loss gate depends on current PET.

| Model | Notation | Description | MCP_Zoo class |
|---|---|---|---|
| M1 | `MC{Oκ Lκ}` | Constant output + constant loss | `MCPBRNN_constant_OutLoss` |
| M2 | `MC{Oκ Lσ}` | Constant output + PET-variable loss | `MCPBRNN_Generic_constant_Out_variableLoss` |
| M3 | `MC{Oσ Lκ}` | Storage-variable output + constant loss | `MCPBRNN_Generic_variable_Out_constantLoss` |
| M4 | `MC{Oσ Lσ}` | Storage-variable output + PET-variable loss | `MCPBRNN_Generic_Scaling` |
| M5 | `MC{Oσ Lσ con}` | M4 + PET-constrained loss | `MCPBRNN_Generic_PETconstraint_Scaling` |

Historical checkpoint references:

| Model | Historical checkpoint |
|---|---|
| M1 | `model_epoch42.pt` |
| M2 | `model_epoch468.pt` |
| M3 | `model_epoch4963.pt` |
| M4 | `model_epoch48.pt` |
| M5 | `model_epoch27.pt` |

---

## 3. Precipitation bias-correction models

These models extend M5 with a precipitation-dependent bias-correction gate.

### 3.1 Piecewise-linear correction: `M-IBCorrPL`

```text
MC{Oσ Lσ con B_Lx},  x = 1,...,5
```

Folders:

```text
M-IBCorrPL/
├── M1-IBCorrPL/
├── M2-IBCorrPL/
├── M3-IBCorrPL/
├── M4-IBCorrPL/
└── M5-IBCorrPL/
```

Class:

```python
MCPBRNN_PETconstraint_IBcorrPL_Generic
```

`x` is the number of piecewise-linear correction nodes.

### 3.2 Piecewise-quadratic correction: `M-IBCorrPQ`

```text
MC{Oσ Lσ con B_Qx},  x = 1,...,5
```

Folders:

```text
M-IBCorrPQ/
├── M1-IBCorrPQ/
├── M2-IBCorrPQ/
├── M3-IBCorrPQ/
├── M4-IBCorrPQ/
└── M5-IBCorrPQ/
```

Class:

```python
MCPBRNN_PETconstraint_IBcorrPQ_Generic
```

`x` is the number of piecewise-quadratic correction nodes.

---

## 4. Mass-relaxation models

The four mass-relaxation cases all retain the M5 hydrologic core and add a signed mass-relaxation correction to the storage update.

The generic state update is

```text
S[t+1] = gR[t] S[t] + P[t] - MR_flux[t]
```

with

```text
gR = 1 - gO - gLc
```

### 4.1 MR1 — Regular

Notation:

```text
MC{Oσ Lσ con Mσ R}
```

Class:

```python
MCPBRNN_PETconstraint_MassRelax_Regular
```

MR formulation:

```python
mr_state = (storage / 500 - exp(bias_b0_yrm)) * exp(weight_s_yvm)
mr_raw = sigmoid(weight_r_yvm) * tanh(mr_state)
mr_gate = min(mr_raw, remember_gate)
mr_flux = mr_gate * abs(storage - exp(bias_b0_yrm) * 500)
```

EVAL scaling:

```python
C_MEAN = 543.3166693
C_STD  = 79.61569861
```

Historical checkpoint: `model_epoch19.pt`

### 4.2 MR2 — Independent

Notation:

```text
MC{Oσ Lσ con MI R}
```

Class:

```python
MCPBRNN_PETconstraint_MassRelax_Indepedent
```

> The historical spelling `Indepedent` is intentionally retained for compatibility.

MR formulation:

```python
mr_sign = sign(storage / 500 - exp(bias_b0_yrm))
mr_raw = sigmoid(weight_r_yvm) * mr_sign
mr_gate = min(mr_raw, remember_gate)
mr_flux = mr_gate * abs(storage - exp(bias_b0_yrm) * 500)
```

EVAL scaling:

```python
C_MEAN = 537.1148789
C_STD  = 82.04150564
```

Historical checkpoint: `model_epoch23.pt`

### 4.3 MR3 — Regular Relaxed

Notation:

```text
MC{Oσ Lσ con Mσr R}
```

Class:

```python
MCPBRNN_PETconstraint_MassRelax_Regular_Relaxed
```

This class is renamed from the historical duplicate `MCPBRNN_PETconstraint_MassRelax_Regular` so it can coexist with MR1 in `MCP_Zoo.py`.

MR formulation:

```python
mr_state = (storage / 500 - bias_b0_yrm) * exp(weight_s_yvm)
mr_raw = sigmoid(weight_r_yvm) * tanh(mr_state)
mr_gate = min(mr_raw, remember_gate)
mr_flux = mr_gate * abs(storage - bias_b0_yrm * 500)
```

EVAL scaling:

```python
C_MEAN = 475.0253683
C_STD  = 89.86101728
```

Historical checkpoint: `model_epoch3.pt`

### 4.4 MR4 — Independent Relaxed

Notation:

```text
MC{Oσ Lσ con MIr R}
```

Class:

```python
MCPBRNN_PETconstraint_MassRelax_Independent_Relaxed
```

This class is renamed from the historical duplicate `MCPBRNN_PETconstraint_MassRelax_Indepedent` so it can coexist with MR2.

MR formulation:

```python
mr_sign = sign(storage / 500 - bias_b0_yrm)
mr_raw = sigmoid(weight_r_yvm) * mr_sign
mr_gate = min(mr_raw, remember_gate)
mr_flux = mr_gate * abs(storage - bias_b0_yrm * 500)
```

EVAL scaling:

```python
C_MEAN = 557.1992146
C_STD  = 85.30399231
```

Historical checkpoint: `model_epoch19.pt`

### MR summary

| Case | Response | Equilibrium parameterization | Parameters | EVAL scaling mean/std |
|---|---|---|---:|---|
| MR1 | Continuous `tanh` | `exp(bias_b0_yrm)` | 10 | `543.3166693 / 79.61569861` |
| MR2 | Sign-based | `exp(bias_b0_yrm)` | 9 | `537.1148789 / 82.04150564` |
| MR3 | Continuous `tanh` | direct `bias_b0_yrm` | 10 | `475.0253683 / 89.86101728` |
| MR4 | Sign-based | direct `bias_b0_yrm` | 9 | `557.1992146 / 85.30399231` |

**Important:** for MR1–MR4, the cleaned training and EVAL scripts use the **scaling mean/std from the corresponding historical EVAL script**.

---

## 5. High-dimensional ANN gate models: `M-ComplexGate`

`A_x` denotes a one-hidden-layer ANN with `x = 1,...,5` hidden nodes.

### 5.1 LossGateOnly

```text
MC{Oσ L_Ax con},  x = 1,...,5
```

Folder structure:

```text
M-ComplexGate/
└── LossGateOnly/
    ├── loss-dim-1/
    ├── loss-dim-2/
    ├── loss-dim-3/
    ├── loss-dim-4/
    └── loss-dim-5/
```

Class:

```python
MCPBRNN_Generic_LossANNGate_PETconstraint
```

The output gate is storage-dependent sigmoid. The loss gate is a PET-driven ANN.

### 5.2 OutputGateOnly

```text
MC{O_Ax Lσ con},  x = 1,...,5
```

Folder structure:

```text
M-ComplexGate/
└── OutputGateOnly/
    ├── output-dim-1/
    ├── output-dim-2/
    ├── output-dim-3/
    ├── output-dim-4/
    └── output-dim-5/
```

Class:

```python
MCPBRNN_Generic_OutputANNGate_PETconstraint
```

The output gate is a storage-driven ANN. The loss gate remains PET-dependent sigmoid.

### 5.3 BothLossOutputGate

```text
MC{O_Ax L_Ay con},  x,y = 1,...,5
```

There are 25 combinations:

```text
loss-dim-1-out-dim-1
...
loss-dim-5-out-dim-5
```

Class:

```python
MCPBRNN_Generic_ANNGate_PETconstraint_Generic
```

The output and loss gates each use a one-hidden-layer ANN.

---

## 6. Multi-information ANN models: `M-MI-ComplexGate`

`MI` means that a gate uses additional information beyond its standard current-time input.

For the current implementation:

- the **output gate** can use current storage plus lagged storage (`t-1`);
- the **loss gate** can additionally use current storage information;
- `+` in the notation denotes the MI extension.

### 6.1 LossGateOnly

```text
MC{Oσ L_Ax+ con},  x = 1,...,5
```

Class:

```python
MCPBRNN_Generic_PETconstraint_MIloss
```

### 6.2 OutputGateOnly

```text
MC{O_Ax+ Lσ con},  x = 1,...,5
```

Class:

```python
MCPBRNN_Generic_PETconstraint_MIoutput
```

### 6.3 BothLossOutputGate

```text
MC{O_Ax+ L_Ay+ con},  x,y = 1,...,5
```

There are 25 combinations.

Class:

```python
MCPBRNN_Generic_PETconstraint_MIoutputloss
```

### 6.4 MI-Only sigmoid models

These cases use direct sigmoid gates instead of ANN hidden layers.

| Folder | Notation | Class |
|---|---|---|
| `loss-gate-only/` | `MC{Oσ Lσ+ con}` | `MCPBRNN_Generic_PETconstraint_MIloss_Sigmoid` |
| `output-gate-only/` | `MC{Oσ+ Lσ con}` | `MCPBRNN_Generic_PETconstraint_MIoutput_Sigmoid` |
| `loss-output-gate/` | `MC{Oσ+ Lσ+ con}` | `MCPBRNN_Generic_PETconstraint_MIoutputloss_Sigmoid` |

---

## 7. Recommended repository structure

```text
project/
├── 20220527-MDUPLEX-LeafRiver/
├── MCPBRNN_lib_tools/
│   ├── MCP_Zoo.py
│   ├── Eval_Metric.py
│   └── Loss_Function.py
│
├── M1/
├── M2/
├── M3/
├── M4/
├── M5/
│
├── M-IBCorrPL/
│   ├── M1-IBCorrPL/
│   └── ...
│
├── M-IBCorrPQ/
│   ├── M1-IBCorrPQ/
│   └── ...
│
├── M-MassRelaxation/
│   ├── MR1/
│   ├── MR2/
│   ├── MR3/
│   └── MR4/
│
├── M-ComplexGate/
│   ├── LossGateOnly/
│   ├── OutputGateOnly/
│   └── BothLossOutputGate/
│
└── M-MI-ComplexGate/
    ├── LossGateOnly/
    ├── OutputGateOnly/
    ├── BothLossOutputGate/
    └── MI-Only/
        ├── loss-gate-only/
        ├── output-gate-only/
        └── loss-output-gate/
```

All model classes should be consolidated in:

```text
MCPBRNN_lib_tools/MCP_Zoo.py
```

---

## 8. Checkpoint convention

For cleaned scripts, the preferred convention is:

```text
<model-folder>/
├── <one checkpoint>.pt
├── training script
└── EVAL script
```

If `--checkpoint` is not supplied, the script searches the relevant model folder for exactly one `.pt` file.

Training workflow:

```text
existing checkpoint
    ↓
strict load
    ↓
continue training
    ↓
evaluate every epoch
    ↓
select best epoch by selection KGE
    ↓
best_model_epochX.pt
```

EVAL workflow:

```text
checkpoint
    ↓
strict load
    ↓
forward-only evaluation
    ↓
metrics + diagnostic time series
```

---

## 9. Common training settings

Unless a case explicitly differs:

```python
time_lag = 0
learning_rate = 0.025
learning_rates = {300: 0.0125, 600: 0.0125}
```

Data split flags:

| Flag | Split |
|---:|---|
| `-99999` | spinup |
| `-1` | training |
| `0` | selection |
| `1` | testing |

Reported metrics:

```text
NSE
KGE
KGE-A
KGE-B
Corr
MSE / mse
KGEss
```

Training diagnostics also include KGE at lags 1, 2, and 3.

---

## 10. Time-series output convention

Cleaned output files use:

```text
time, <result variable>, phase
```

with phases:

```text
spinup1
spinup2
spinup3
simulation
```

The time axis contains:

- three repetitions of WY1949 (`1948-10-01` to `1949-09-30`) for spinup;
- formal simulation from `1948-10-01` through `1988-09-30`;
- total length: **15,705 rows**.

Repeated calendar dates across spinup phases are intentional; `phase` distinguishes them.

---

## 11. Implementation conventions

Cleaned model classes follow these rules:

- preserve historical checkpoint parameter names and tensor shapes;
- preserve pre-update versus post-update hydrologic timing;
- use **remember gate** terminology for `gR`;
- use `x.new_zeros(...)` for device-safe output tensors;
- keep the `seq_len = 1` assumption explicit;
- remove unused imports, dead prototype code, and obsolete initialization logic;
- keep equations compact and readable;
- do not rename `state_dict` parameter keys;
- Python class names may be renamed only when necessary to avoid duplicate class names in `MCP_Zoo.py`.

For MR3 and MR4, only the Python class names were disambiguated; the checkpoint parameter keys remain unchanged.

---

## 12. Environment and dependencies

The repository was organized for the following Python environment.

### `requirement.txt`

```text
numpy==2.2.2
pandas==2.2.3
torch==2.9.0
torchvision==0.24.0
tqdm==4.67.1
scikit-learn==1.6.1
```

Install with:

```bash
pip install -r requirement.txt
```

### `environment.yml`

```yaml
name: MCP-SingleNode-ModelZoo
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - notebook
  - pip:
      - numpy==2.2.2
      - pandas==2.2.3
      - torch==2.9.0
      - torchvision==0.24.0
      - tqdm==4.67.1
      - scikit-learn==1.6.1
```

Create and activate the Conda environment with:

```bash
conda env create -f environment.yml
conda activate MCP-SingleNode-ModelZoo
```

