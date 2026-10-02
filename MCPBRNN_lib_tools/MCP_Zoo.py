import torch
import torch.nn as nn
from torch import Tensor

class MCPBRNN_constant_OutLoss(nn.Module):
    """Single-node MCP with constant output, loss, and remember gates.

    The three learned gate parameters partition stored water among discharge,
    loss, and remembered storage. The input gate/bypass is fixed to zero, so
    all precipitation enters the storage.

    Parameter names and shapes are preserved so original M1 checkpoints load
    without modification.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained in the constructor for consistency with the MCP model zoo.
        del input_size, gate_dim, spinLen, traintimeLen, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("M1 is a single-node MCP and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Parameter names/shapes intentionally match the original implementation.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) parameter initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag):
        del epoch  # Retained for consistency with other MCP implementations.

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError("M1 expects seq_length == 1.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states/fluxes and gate diagnostics.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)
        loss = x.new_zeros(batch_size, hidden_size)
        bypass = x.new_zeros(batch_size, hidden_size)

        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_output = x.new_zeros(batch_size, hidden_size)
        gate_loss = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        # Constant mass-partition gates.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = exp_output + exp_loss + exp_remember

        output_fraction = exp_output / gate_sum
        loss_fraction = exp_loss / gate_sum

        # Preserve the original M1 formulation exactly:
        # gR = 1 - gO - gL
        remember_fraction = (1.0 - output_fraction - loss_fraction)

        for b in range(time_lag, batch_size):

            precipitation = x[0, b, 0].reshape(1, 1)

            # Save pre-update storage and fluxes,
            # matching the original M1 timing.
            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss[b, :] = (loss_fraction * storage)[0]

            gate_output[b, :] = output_fraction[0]
            gate_loss[b, :] = loss_fraction[0]
            gate_remember[b, :] = remember_fraction[0]

            # Input bypass is zero:
            # all precipitation enters storage.
            storage = (remember_fraction * storage + precipitation)

        return (discharge, storage_series, loss, bypass, gate_input, gate_output, gate_loss, gate_remember, )


class MCPBRNN_Generic_constant_Out_variableLoss(nn.Module):
    """Single-node MCP with a constant output gate and PET-dependent loss gate.

    The output gate is constant in time, while the loss gate varies with PET.
    The remember gate is defined by mass conservation as
    ``gR = 1 - gO - gL``. The input gate/bypass is fixed to zero, so all
    precipitation enters storage.

    Parameter names and shapes are preserved so original M2 checkpoints load
    without modification.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained in the constructor for consistency with the MCP model zoo.
        del input_size, gate_dim, spinLen, traintimeLen, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("M2 is a single-node MCP and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Parameter names/shapes intentionally match the original implementation.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) parameter initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag):
        del epoch  # Retained for consistency with other MCP implementations.

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("M2 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("M2 requires precipitation and PET as input features.")

        hidden_size = self.hidden_size
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states/fluxes and gate diagnostics.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)
        loss = x.new_zeros(batch_size, hidden_size)
        bypass = x.new_zeros(batch_size, hidden_size)

        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_output = x.new_zeros(batch_size, hidden_size)
        gate_loss = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET normalization constants used by the original M2 formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Constant scaling terms shared by the output/loss gates.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = exp_output / gate_sum
        loss_scale = exp_loss / gate_sum

        for b in range(time_lag, batch_size):

            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            # Constant output gate.
            output_fraction = output_scale

            # PET-dependent loss gate.
            loss_fraction = loss_scale * torch.sigmoid(
                self.bias_b0_ylm.unsqueeze(0)
                + ((pet - pet_mean) / pet_std)
                * self.weight_b2_ylm
            )

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction)

            # Save pre-update storage and fluxes,
            # matching the original M2 timing.
            storage_series[b, :] = storage[0]

            discharge[b, :] = (output_fraction * storage)[0]

            loss[b, :] = (loss_fraction * storage)[0]

            gate_output[b, :] = output_fraction[0]
            gate_loss[b, :] = loss_fraction[0]
            gate_remember[b, :] = remember_fraction[0]

            # Input bypass is zero:
            # all precipitation enters storage.
            storage = (remember_fraction * storage + precipitation)

        return (discharge, storage_series, loss, bypass, gate_input, gate_output, gate_loss, gate_remember, )

class MCPBRNN_Generic_variable_Out_constantLoss(nn.Module):
    """Single-node MCP with a storage-dependent output gate
    and a constant loss gate.

    The output gate varies with the current storage state.
    The loss gate is constant in time. The remember gate is
    determined by mass conservation:

        gR = 1 - gO - gL

    The input gate is fixed to zero, so all precipitation
    enters storage.

    Parameter names and shapes are preserved for compatibility
    with the original M3 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MCP Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("M3 is a single-node MCP and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Keep original parameter names and shapes for checkpoint compatibility.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)
            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std, ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if seq_len != 1:
            raise ValueError("M3 expects seq_length == 1.")

        hidden_size = self.hidden_size

        # Initial storage.
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)
        loss = x.new_zeros(batch_size, hidden_size)
        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_output = x.new_zeros(batch_size, hidden_size)
        gate_loss = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        # Base mass-partition terms.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_fraction = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        for b in range(time_lag, batch_size):

            precipitation = (x[0, b, 0].reshape(1, 1))

            # Storage-dependent output gate.
            output_state = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # Preserve the original M3 formulation exactly.
            remember_fraction = (1.0 - output_fraction - loss_fraction)

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]

            discharge[b, :] = (output_fraction * storage)[0]

            loss[b, :] = (loss_fraction * storage)[0]

            # Save gates.
            gate_output[b, :] = (output_fraction[0])
            gate_loss[b, :] = (loss_fraction[0])
            gate_remember[b, :] = (remember_fraction[0])

            # Input gate = 0:
            # all precipitation enters storage.
            storage = (remember_fraction * storage + precipitation)

        return (discharge, storage_series, loss, bypass, gate_input, gate_output, gate_loss, gate_remember, )

class MCPBRNN_Generic_Scaling(nn.Module):
    """Single-node MCP with storage-dependent output
    and PET-dependent loss gates.

    The output gate varies with normalized storage.
    The loss gate varies with normalized PET.
    The remember gate is defined by mass conservation:

        gR = 1 - gO - gL

    The input gate is fixed to zero, so all precipitation
    enters storage.

    Parameter names and shapes are preserved for compatibility
    with the original M4 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("M4 is a single-node MCP and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names/shapes.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std, ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("M4 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("M4 requires precipitation and PET.")

        hidden_size = self.hidden_size

        # Initial storage.
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)
        loss = x.new_zeros(batch_size, hidden_size)
        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_output = x.new_zeros(batch_size, hidden_size)
        gate_loss = x.new_zeros(batch_size, hidden_size)
        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET scaling from the original M4 formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition terms.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        loss_bias = (self.bias_b0_ylm.unsqueeze(0))

        for b in range(time_lag, batch_size):

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # Storage-dependent output gate.
            output_state = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # PET-dependent loss gate.
            loss_state = torch.addmm(loss_bias, (pet - pet_mean) / pet_std, self.weight_b2_ylm, )

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction)

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]

            discharge[b, :] = (output_fraction * storage)[0]

            loss[b, :] = (loss_fraction * storage)[0]

            # Save gate diagnostics.
            gate_output[b, :] = (output_fraction[0])

            gate_loss[b, :] = (loss_fraction[0])

            gate_remember[b, :] = (remember_fraction[0])

            # Input bypass = 0:
            # all precipitation enters storage.
            storage = (remember_fraction * storage + precipitation)

        return (discharge, storage_series, loss, bypass, gate_input, gate_output, gate_loss, gate_remember, )

class MCPBRNN_Generic_PETconstraint_Scaling(nn.Module):
    """Single-node MCP with scaled output/loss gates
    and a PET-constrained loss flux.

    The output gate varies with normalized storage.
    The unconstrained loss gate varies with normalized PET.

    The actual loss gate is constrained so that the loss flux
    cannot exceed available PET:

        gLc = min(gL, PET / storage)

    for positive storage.

    The remember gate is then defined by mass conservation:

        gR = 1 - gO - gLc

    The input gate/bypass is fixed to zero, so all precipitation
    enters storage.

    Parameter names and shapes are preserved for compatibility
    with the original M5 checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for a consistent MCP Zoo interface.
        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("M5 is a single-node MCP and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Preserve original parameter names and shapes.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std, ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("M5 expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("M5 requires precipitation and PET.")

        hidden_size = self.hidden_size

        # Initial storage.
        storage = x.new_zeros(1, hidden_size)

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size)
        storage_series = x.new_zeros(batch_size, hidden_size)

        loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        loss_constrained = x.new_zeros(batch_size, hidden_size)

        bypass = x.new_zeros(batch_size, hidden_size)

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size)
        gate_output = x.new_zeros(batch_size, hidden_size)

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size)
        gate_loss_constrained = x.new_zeros(batch_size, hidden_size)

        gate_remember = x.new_zeros(batch_size, hidden_size)

        # PET scaling from the original formulation.
        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition terms.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        loss_bias = (self.bias_b0_ylm.unsqueeze(0))

        for b in range(time_lag, batch_size):

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # Storage-dependent output gate.
            output_state = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # PET-dependent unconstrained loss gate.
            loss_state = torch.addmm(loss_bias, (pet - pet_mean) / pet_std, self.weight_b2_ylm, )

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # PET constraint:
            # actual loss cannot exceed PET.
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - self.relu_l(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate uses the constrained loss gate.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # Save pre-update storage and fluxes.
            storage_series[b, :] = storage[0]

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            # Save gate diagnostics.
            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            # Input bypass = 0:
            # all precipitation enters storage.
            storage = (remember_fraction * storage + precipitation)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )


class MCPBRNN_PETconstraint_IBcorrPL_Generic(nn.Module):
    """Single-node MCP with piecewise-linear precipitation bias correction.

    Precipitation is corrected using a sum of ReLU hinge functions.
    The hydrologic core uses a storage-dependent output gate,
    a PET-dependent loss gate, and a PET-constrained loss flux.

    The remember gate is defined by mass conservation as

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and shapes are preserved for compatibility
    with the original IBCorrPL checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        gate_dim_ucorr: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("IBCorrPL is a single-node MCP and requires hidden_size == 1.")

        if gate_dim_ucorr < 1:
            raise ValueError("gate_dim_ucorr must be at least 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim_ucorr = gate_dim_ucorr

        # Hydrologic-core parameters.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Piecewise-linear precipitation-correction parameters.
        self.ln_wj = nn.Parameter(torch.empty(gate_dim_ucorr, hidden_size, ))

        self.relu_bj = nn.Parameter(torch.empty(1, gate_dim_ucorr, ))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization order."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.ln_wj.uniform_(0.0, 1.0)
            self.relu_bj.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, c_mean, c_std, ):
        del epoch

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("IBCorrPL expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("IBCorrPL requires precipitation and PET.")

        hidden_size = self.hidden_size

        storage = x.new_zeros(1, hidden_size, )

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size, )

        storage_series = x.new_zeros(batch_size, hidden_size, )

        loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        loss_constrained = x.new_zeros(batch_size, hidden_size, )

        bypass = x.new_zeros(batch_size, hidden_size, )

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size, )

        gate_output = x.new_zeros(batch_size, hidden_size, )

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        gate_loss_constrained = x.new_zeros(batch_size, hidden_size, )

        gate_remember = x.new_zeros(batch_size, hidden_size, )

        bias_correction = x.new_zeros(batch_size, hidden_size, )

        # Original scaling constants.
        precipitation_max = 221.5190
        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition terms.
        exp_output = torch.exp(self.weight_r_yom)

        exp_loss = torch.exp(self.weight_r_ylm)

        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        loss_bias = (self.bias_b0_ylm.unsqueeze(0))

        for b in range(time_lag, batch_size, ):

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # --------------------------------------------------
            # Piecewise-linear precipitation bias correction
            # --------------------------------------------------
            normalized_precipitation = (precipitation.expand(1, self.gate_dim_ucorr, ) / precipitation_max)

            hinge_response = torch.relu(normalized_precipitation - torch.relu(self.relu_bj))

            correction = torch.mm(hinge_response, self.ln_wj, )

            corrected_precipitation = (precipitation + correction * precipitation_max)

            # --------------------------------------------------
            # Storage-dependent output gate
            # --------------------------------------------------
            output_state = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # --------------------------------------------------
            # PET-dependent loss gate
            # --------------------------------------------------
            loss_state = torch.addmm(loss_bias, (pet - pet_mean) / pet_std, self.weight_b2_ylm, )

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # PET constraint.
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - torch.relu(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # --------------------------------------------------
            # Save pre-update state/flux diagnostics
            # --------------------------------------------------
            storage_series[b, :] = (storage[0])

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            bias_correction[b, :] = (correction[0])

            # Input gate/bypass = 0:
            # corrected precipitation enters storage directly.
            storage = (remember_fraction * storage + corrected_precipitation)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            bias_correction,
        )

class MCPBRNN_PETconstraint_IBcorrPQ_Generic(nn.Module):
    """Single-node MCP with piecewise-quadratic precipitation bias correction.

    The precipitation correction uses a piecewise-linear multiplier, making
    corrected precipitation piecewise quadratic with respect to precipitation.

    The hydrologic core uses a storage-dependent output gate, a PET-dependent
    loss gate, and a PET-constrained loss flux.

    The remember gate is

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and shapes are preserved for compatibility with the
    original IBCorrPQ checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim: int,
        gate_dim_ucorr: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, gate_dim, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("IBCorrPQ is a single-node MCP and requires hidden_size == 1.")

        if gate_dim_ucorr < 1:
            raise ValueError("gate_dim_ucorr must be at least 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim_ucorr = gate_dim_ucorr

        # Hydrologic-core parameters.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Piecewise-quadratic precipitation correction.
        self.ln_b = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.ln_wj = nn.Parameter(torch.empty(gate_dim_ucorr, hidden_size, ))

        self.relu_bj = nn.Parameter(torch.empty(1, gate_dim_ucorr, ))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization order."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.ln_b.uniform_(0.0, 1.0)
            self.ln_wj.uniform_(0.0, 1.0)
            self.relu_bj.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, ):
        # Retained for compatibility with historical call signatures.
        del epoch, y_obs

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("IBCorrPQ expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("IBCorrPQ requires precipitation and PET.")

        hidden_size = self.hidden_size

        storage = x.new_zeros(1, hidden_size, )

        discharge = x.new_zeros(batch_size, hidden_size, )

        storage_series = x.new_zeros(batch_size, hidden_size, )

        loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        loss_constrained = x.new_zeros(batch_size, hidden_size, )

        bypass = x.new_zeros(batch_size, hidden_size, )

        gate_input = x.new_zeros(batch_size, hidden_size, )

        gate_output = x.new_zeros(batch_size, hidden_size, )

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        gate_loss_constrained = x.new_zeros(batch_size, hidden_size, )

        gate_remember = x.new_zeros(batch_size, hidden_size, )

        precipitation_corrected = x.new_zeros(batch_size, hidden_size, )

        precipitation_max = 221.5190
        pet_mean = 2.9086
        pet_std = 1.8980

        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        loss_bias = (self.bias_b0_ylm.unsqueeze(0))

        for b in range(time_lag, batch_size, ):

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # ---------------------------------------------
            # Piecewise-quadratic precipitation correction
            # ---------------------------------------------
            normalized_precipitation = (precipitation.expand(1, self.gate_dim_ucorr, ) / precipitation_max)

            hinge_response = torch.relu(normalized_precipitation - torch.relu(self.relu_bj))

            correction_factor = torch.addmm(self.ln_b, hinge_response, self.ln_wj, )

            corrected_precipitation = (precipitation * correction_factor)

            # ---------------------------------------------
            # Storage-dependent output gate
            # ---------------------------------------------
            output_state = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # ---------------------------------------------
            # PET-dependent loss gate
            # ---------------------------------------------
            loss_state = torch.addmm(loss_bias, (pet - pet_mean) / pet_std, self.weight_b2_ylm, )

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # PET constraint.
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - torch.relu(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # Pre-update diagnostics.
            storage_series[b, :] = (storage[0])

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            precipitation_corrected[b, :] = (corrected_precipitation[0])

            # Corrected precipitation enters storage directly.
            storage = (remember_fraction * storage + corrected_precipitation)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            precipitation_corrected,
        )

class MCPBRNN_Generic_PETconstraint_MIloss(nn.Module):
    """Single-node MCP with a multi-dimensional loss-gate ANN.

    The output gate uses one storage-dependent unit.
    The loss gate uses a hidden layer of dimension ``gate_dim_l``
    driven jointly by normalized storage and PET.

    The unconstrained loss gate is subsequently limited by PET.
    The remember gate is defined by mass conservation as

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and shapes are preserved for compatibility
    with the original LossGateOnly checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim_o: int,
        gate_dim_l: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("LossGateOnly is a single-node MCP " "and requires hidden_size == 1.")

        if gate_dim_o != 1:
            raise ValueError("LossGateOnly requires gate_dim_o == 1.")

        if gate_dim_l < 1:
            raise ValueError("gate_dim_l must be at least 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim_o = gate_dim_o
        self.gate_dim_l = gate_dim_l

        # Base mass-partition parameters.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Output-gate parameters.
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))

        # Multi-dimensional loss-gate parameters.
        self.weight_b1_ylm = nn.Parameter(torch.empty(2, gate_dim_l))
        self.bias_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))
        self.bias_ln_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.relu_bias_l = nn.Parameter(torch.empty(1, gate_dim_l))

        # Original activation functions.
        self.relu_l = nn.SELU()
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization order."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.bias_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.bias_ln_ylm.uniform_(0.0, 1.0)
            self.relu_bias_l.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, ):
        # Retained for compatibility with the historical interface.
        del epoch, y_obs

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("LossGateOnly expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("LossGateOnly requires precipitation and PET.")

        hidden_size = self.hidden_size

        storage = x.new_zeros(1, hidden_size, )

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size, )

        storage_series = x.new_zeros(batch_size, hidden_size, )

        loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        loss_constrained = x.new_zeros(batch_size, hidden_size, )

        bypass = x.new_zeros(batch_size, hidden_size, )

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size, )

        gate_output = x.new_zeros(batch_size, hidden_size, )

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        gate_loss_constrained = x.new_zeros(batch_size, hidden_size, )

        gate_remember = x.new_zeros(batch_size, hidden_size, )

        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition scaling.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        loss_hidden_bias = (self.bias_ylm.unsqueeze(0).expand(1, self.gate_dim_l, ))

        for b in range(time_lag, batch_size, ):

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # ---------------------------------------------
            # Storage-dependent output gate
            # ---------------------------------------------
            output_state = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # ---------------------------------------------
            # Multi-dimensional loss-gate ANN
            # Inputs: normalized storage and PET
            # ---------------------------------------------
            loss_features = torch.cat(((storage - c_mean) / c_std, (pet - pet_mean) / pet_std, ), dim=1, )

            loss_hidden_state = torch.addmm(loss_hidden_bias, loss_features, self.weight_b1_ylm, )

            # Preserve original SELU activation and learned thresholds.
            loss_hidden = self.relu_l(loss_hidden_state - self.relu_bias_l)

            loss_state = torch.addmm(self.bias_ln_ylm, loss_hidden, self.weight_b2_ylm, )

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # ---------------------------------------------
            # PET constraint
            # ---------------------------------------------
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - self.relu(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # ---------------------------------------------
            # Pre-update diagnostics
            # ---------------------------------------------
            storage_series[b, :] = (storage[0])

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            # Input gate/bypass = 0:
            # all precipitation enters storage.
            storage = (remember_fraction * storage + precipitation)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )

class MCPBRNN_Generic_PETconstraint_MIoutput(nn.Module):
    """Single-node MCP with a multi-dimensional output-gate ANN.

    The output gate uses current and lagged storage as its two inputs.
    The hidden-layer dimension is controlled by ``gate_dim_o``.

    The loss gate remains a one-dimensional PET-dependent gate and is
    subsequently constrained so that loss cannot exceed available PET.

    The remember gate is defined by mass conservation as

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and shapes are preserved for compatibility with the
    original OutputGateOnly checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim_o: int,
        gate_dim_l: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("OutputGateOnly is a single-node MCP " "and requires hidden_size == 1.")

        if gate_dim_o < 1:
            raise ValueError("gate_dim_o must be at least 1.")

        if gate_dim_l != 1:
            raise ValueError("OutputGateOnly requires gate_dim_l == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim_o = gate_dim_o
        self.gate_dim_l = gate_dim_l

        # Base mass-partition parameters.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Multi-dimensional output-gate parameters.
        self.bias_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(2, gate_dim_o))
        self.weight_b2_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))
        self.relu_bias_o = nn.Parameter(torch.empty(1, gate_dim_o))
        self.bias_ln_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # One-dimensional PET-dependent loss gate.
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))

        # Preserve original activation functions.
        self.relu_o = nn.SELU()
        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization order."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.weight_b2_yom.uniform_(0.0, 1.0)
            self.relu_bias_o.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

            self.bias_ln_yom.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, ):
        # Retained for compatibility with historical call signatures.
        del epoch, y_obs

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("OutputGateOnly expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("OutputGateOnly requires precipitation and PET.")

        hidden_size = self.hidden_size

        # Current and lagged storage states.
        storage = x.new_zeros(1, hidden_size, )

        previous_storage = x.new_zeros(1, hidden_size, )

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size, )

        storage_series = x.new_zeros(batch_size, hidden_size, )

        loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        loss_constrained = x.new_zeros(batch_size, hidden_size, )

        bypass = x.new_zeros(batch_size, hidden_size, )

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size, )

        gate_output = x.new_zeros(batch_size, hidden_size, )

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        gate_loss_constrained = x.new_zeros(batch_size, hidden_size, )

        gate_remember = x.new_zeros(batch_size, hidden_size, )

        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition scaling.
        exp_output = torch.exp(self.weight_r_yom)

        exp_loss = torch.exp(self.weight_r_ylm)

        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_hidden_bias = (self.bias_yom.unsqueeze(0).expand(1, self.gate_dim_o, ))

        loss_bias = (self.bias_b0_ylm.unsqueeze(0).expand(1, self.gate_dim_l, ))

        for b in range(batch_size):

            # Preserve the original lag-state initialization/update logic.
            if b >= time_lag:
                lagged_storage = previous_storage
            else:
                lagged_storage = x.new_zeros(1, hidden_size, )

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # --------------------------------------------------
            # Multi-dimensional output-gate ANN
            #
            # Inputs:
            #   1. current storage
            #   2. lagged storage
            # --------------------------------------------------
            output_features = torch.cat(((storage - c_mean) / c_std, (lagged_storage - c_mean) / c_std, ), dim=1, )

            output_hidden_state = torch.addmm(output_hidden_bias, output_features, self.weight_b1_yom, )

            output_hidden = self.relu_o(output_hidden_state - self.relu_bias_o)

            output_state = torch.addmm(self.bias_ln_yom, output_hidden, self.weight_b2_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # --------------------------------------------------
            # One-dimensional PET-dependent loss gate
            # --------------------------------------------------
            loss_state = torch.addmm(loss_bias, (pet - pet_mean) / pet_std, self.weight_b2_ylm, )

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # PET constraint.
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - self.relu_l(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # --------------------------------------------------
            # Pre-update diagnostics
            # --------------------------------------------------
            storage_series[b, :] = (storage[0])

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            # Preserve the pre-update storage for the lagged
            # output-gate input, matching the original implementation.
            current_preupdate_storage = storage

            # Input gate/bypass = 0:
            # all precipitation enters storage.
            storage = (remember_fraction * storage + precipitation)

            if b >= time_lag:
                previous_storage = (current_preupdate_storage)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )

class MCPBRNN_Generic_PETconstraint_MIoutputloss(nn.Module):
    """Single-node MCP with multi-input ANN output and loss gates.

    The output gate uses current and lagged storage as inputs.
    The loss gate uses current storage and PET as inputs.

    Both gates use SELU hidden layers whose dimensions are controlled by
    ``gate_dim_o`` and ``gate_dim_l``.

    The loss gate is constrained so that actual loss cannot exceed PET.
    The remember gate is defined by mass conservation:

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and shapes are preserved for compatibility with the
    original BothLossOutputGate checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim_o: int,
        gate_dim_l: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("BothLossOutputGate is a single-node MCP " "and requires hidden_size == 1.")

        if gate_dim_o < 1:
            raise ValueError("gate_dim_o must be at least 1.")

        if gate_dim_l < 1:
            raise ValueError("gate_dim_l must be at least 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim_o = gate_dim_o
        self.gate_dim_l = gate_dim_l

        # Base mass-partition parameters.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Multi-input output-gate ANN.
        self.bias_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(2, gate_dim_o))
        self.weight_b2_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))
        self.relu_bias_o = nn.Parameter(torch.empty(1, gate_dim_o))
        self.bias_ln_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Multi-input loss-gate ANN.
        self.bias_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.empty(2, gate_dim_l))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))
        self.relu_bias_l = nn.Parameter(torch.empty(1, gate_dim_l))
        self.bias_ln_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_o = nn.SELU()
        self.relu_l = nn.SELU()
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization order."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.weight_b2_yom.uniform_(0.0, 1.0)
            self.relu_bias_o.uniform_(0.0, 1.0)
            self.bias_ln_yom.uniform_(0.0, 1.0)

            self.bias_ylm.uniform_(0.0, 1.0)
            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)
            self.relu_bias_l.uniform_(0.0, 1.0)
            self.bias_ln_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, ):
        # Retained for compatibility with historical scripts.
        del epoch, y_obs

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("BothLossOutputGate expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("BothLossOutputGate requires precipitation and PET.")

        hidden_size = self.hidden_size

        # Current and lagged storage states.
        storage = x.new_zeros(1, hidden_size, )

        previous_storage = x.new_zeros(1, hidden_size, )

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size, )

        storage_series = x.new_zeros(batch_size, hidden_size, )

        loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        loss_constrained = x.new_zeros(batch_size, hidden_size, )

        bypass = x.new_zeros(batch_size, hidden_size, )

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size, )

        gate_output = x.new_zeros(batch_size, hidden_size, )

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        gate_loss_constrained = x.new_zeros(batch_size, hidden_size, )

        gate_remember = x.new_zeros(batch_size, hidden_size, )

        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition scaling.
        exp_output = torch.exp(self.weight_r_yom)
        exp_loss = torch.exp(self.weight_r_ylm)
        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_hidden_bias = (self.bias_yom.unsqueeze(0).expand(1, self.gate_dim_o, ))

        loss_hidden_bias = (self.bias_ylm.unsqueeze(0).expand(1, self.gate_dim_l, ))

        for b in range(batch_size):

            # Preserve original lagged-storage behavior.
            if b >= time_lag:
                lagged_storage = (previous_storage)
            else:
                lagged_storage = (x.new_zeros(1, hidden_size, ))

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # --------------------------------------------------
            # Multi-input output gate
            #
            # Inputs:
            #   current storage
            #   lagged storage
            # --------------------------------------------------
            output_features = torch.cat(((storage - c_mean) / c_std, (lagged_storage - c_mean) / c_std, ), dim=1, )

            output_hidden_state = (torch.addmm(output_hidden_bias, output_features, self.weight_b1_yom, ))

            output_hidden = self.relu_o(output_hidden_state - self.relu_bias_o)

            output_state = torch.addmm(self.bias_ln_yom, output_hidden, self.weight_b2_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # --------------------------------------------------
            # Multi-input loss gate
            #
            # Inputs:
            #   current storage
            #   PET
            # --------------------------------------------------
            loss_features = torch.cat(((storage - c_mean) / c_std, (pet - pet_mean) / pet_std, ), dim=1, )

            loss_hidden_state = (torch.addmm(loss_hidden_bias, loss_features, self.weight_b1_ylm, ))

            loss_hidden = self.relu_l(loss_hidden_state - self.relu_bias_l)

            loss_state = torch.addmm(self.bias_ln_ylm, loss_hidden, self.weight_b2_ylm, )

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # PET constraint.
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - self.relu(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # Pre-update diagnostics.
            storage_series[b, :] = (storage[0])

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            current_preupdate_storage = (storage)

            # Input gate/bypass = 0.
            storage = (remember_fraction * storage + precipitation)

            if b >= time_lag:
                previous_storage = (current_preupdate_storage)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )

class MCPBRNN_Generic_PETconstraint_MIloss_Sigmoid(nn.Module):
    """Single-node MCP with a direct multi-input sigmoid loss gate.

    The output gate depends on normalized storage.

    The loss gate directly combines normalized storage and PET:

        gL = scale_L * sigmoid(
            bias_L
            + storage_term
            + PET_term
        )

    The loss gate is then constrained so that loss cannot exceed PET.

    The remember gate is defined by mass conservation:

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and shapes are preserved for compatibility with the
    original MI-Only loss-gate checkpoint.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim_o: int,
        gate_dim_l: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        # Retained only for compatibility with the historical interface.
        del input_size, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MI-Only loss-gate model requires hidden_size == 1.")

        # This historical formulation is dimension-1 only.
        if gate_dim_o != 1:
            raise ValueError("MI-Only loss-gate model requires gate_dim_o == 1.")

        if gate_dim_l != 1:
            raise ValueError("MI-Only loss-gate model requires gate_dim_l == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Base mass-partition parameters.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Storage-dependent output gate.
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))

        # Direct storage + PET loss gate.
        self.weight_b1_ylm = nn.Parameter(torch.empty(gate_dim_l, gate_dim_l))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization order."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, ):
        # Retained for compatibility with old calling code.
        del epoch, y_obs

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MI-Only loss-gate model expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("Model requires precipitation and PET.")

        hidden_size = self.hidden_size

        storage = x.new_zeros(1, hidden_size, )

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size, )

        storage_series = x.new_zeros(batch_size, hidden_size, )

        loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        loss_constrained = x.new_zeros(batch_size, hidden_size, )

        bypass = x.new_zeros(batch_size, hidden_size, )

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size, )

        gate_output = x.new_zeros(batch_size, hidden_size, )

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        gate_loss_constrained = x.new_zeros(batch_size, hidden_size, )

        gate_remember = x.new_zeros(batch_size, hidden_size, )

        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition scaling.
        exp_output = torch.exp(self.weight_r_yom)

        exp_loss = torch.exp(self.weight_r_ylm)

        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        loss_bias = (self.bias_b0_ylm.unsqueeze(0))

        for b in range(time_lag, batch_size, ):

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # ---------------------------------------------
            # Storage-dependent output gate
            # ---------------------------------------------
            output_state = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # ---------------------------------------------
            # Direct multi-input sigmoid loss gate
            #
            # Inputs:
            #   normalized storage
            #   normalized PET
            # ---------------------------------------------
            storage_loss_term = torch.addmm(loss_bias, (storage - c_mean) / c_std, self.weight_b1_ylm, )

            pet_loss_term = torch.mm((pet - pet_mean) / pet_std, self.weight_b2_ylm, )

            loss_state = (storage_loss_term + pet_loss_term)

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # PET constraint.
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - torch.relu(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # Pre-update diagnostics.
            storage_series[b, :] = (storage[0])

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            # Input gate/bypass = 0.
            storage = (remember_fraction * storage + precipitation)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )

class MCPBRNN_Generic_PETconstraint_MIoutput_Sigmoid(nn.Module):
    """Single-node MCP with a direct multi-input sigmoid output gate.

    The output gate directly combines normalized current storage
    and normalized lagged storage.

    The loss gate remains PET-dependent and is constrained so that
    the loss flux cannot exceed available PET.

    The remember gate is

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and shapes are preserved for compatibility with
    the original MI-Only output-gate checkpoint.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim_o: int,
        gate_dim_l: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MI-Only output-gate model requires hidden_size == 1.")

        if gate_dim_o != 1:
            raise ValueError("MI-Only output-gate model requires gate_dim_o == 1.")

        if gate_dim_l != 1:
            raise ValueError("MI-Only output-gate model requires gate_dim_l == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Base mass-partition parameters.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Direct current/lagged-storage output gate.
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(gate_dim_o, gate_dim_o))
        self.weight_b2_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))

        # PET-dependent loss gate.
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization order."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)
            self.weight_b2_yom.uniform_(0.0, 1.0)

            self.bias_b0_ylm.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, ):
        # Retained for compatibility with historical calling code.
        del epoch, y_obs

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MI-Only output-gate model expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("Model requires precipitation and PET.")

        hidden_size = self.hidden_size

        storage = x.new_zeros(1, hidden_size, )

        lagged_storage_state = x.new_zeros(1, hidden_size, )

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size, )

        storage_series = x.new_zeros(batch_size, hidden_size, )

        loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        loss_constrained = x.new_zeros(batch_size, hidden_size, )

        bypass = x.new_zeros(batch_size, hidden_size, )

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size, )

        gate_output = x.new_zeros(batch_size, hidden_size, )

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        gate_loss_constrained = x.new_zeros(batch_size, hidden_size, )

        gate_remember = x.new_zeros(batch_size, hidden_size, )

        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition scaling.
        exp_output = torch.exp(self.weight_r_yom)

        exp_loss = torch.exp(self.weight_r_ylm)

        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        loss_bias = (self.bias_b0_ylm.unsqueeze(0))

        for b in range(batch_size):

            if b >= time_lag:
                lagged_storage = (lagged_storage_state)
            else:
                lagged_storage = (x.new_zeros(1, hidden_size, ))

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # ---------------------------------------------
            # Direct current + lagged storage output gate
            # ---------------------------------------------
            current_storage_term = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            lagged_storage_term = torch.mm((lagged_storage - c_mean) / c_std, self.weight_b2_yom, )

            output_state = (current_storage_term + lagged_storage_term)

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # ---------------------------------------------
            # PET-dependent loss gate
            # ---------------------------------------------
            loss_state = torch.addmm(loss_bias, (pet - pet_mean) / pet_std, self.weight_b2_ylm, )

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # PET constraint.
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - torch.relu(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # Pre-update diagnostics.
            storage_series[b, :] = (storage[0])

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            current_preupdate_storage = (storage)

            # Input gate/bypass = 0.
            storage = (remember_fraction * storage + precipitation)

            if b >= time_lag:
                lagged_storage_state = (current_preupdate_storage)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )

class MCPBRNN_Generic_PETconstraint_MIoutputloss_Sigmoid(nn.Module):
    """Single-node MCP with direct multi-input sigmoid output and loss gates.

    The output gate directly combines normalized current storage
    and normalized lagged storage.

    The loss gate directly combines normalized current storage
    and normalized PET.

    The loss gate is constrained so that the loss flux cannot
    exceed available PET.

    The remember gate is

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and shapes are preserved for compatibility
    with the original MI-Only loss-output-gate checkpoint.
    """

    def __init__(
        self,
        input_size: int,
        gate_dim_o: int,
        gate_dim_l: int,
        spinLen: int,
        traintimeLen: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, spinLen, traintimeLen
        del initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MI-Only loss-output-gate model requires hidden_size == 1.")

        if gate_dim_o != 1:
            raise ValueError("MI-Only loss-output-gate model requires gate_dim_o == 1.")

        if gate_dim_l != 1:
            raise ValueError("MI-Only loss-output-gate model requires gate_dim_l == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Base mass-partition parameters.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Direct current/lagged-storage output gate.
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))

        # Direct storage/PET loss gate.
        self.weight_b1_ylm = nn.Parameter(torch.empty(gate_dim_l, gate_dim_l))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Preserve the original U(0, 1) initialization order."""
        with torch.no_grad():
            self.weight_r_yom.uniform_(0.0, 1.0)
            self.weight_r_ylm.uniform_(0.0, 1.0)
            self.weight_r_yfm.uniform_(0.0, 1.0)

            self.bias_b0_yom.uniform_(0.0, 1.0)
            self.weight_b1_yom.uniform_(0.0, 1.0)

            self.weight_b1_ylm.uniform_(0.0, 1.0)
            self.bias_b0_ylm.uniform_(0.0, 1.0)

            self.weight_b2_yom.uniform_(0.0, 1.0)
            self.weight_b2_ylm.uniform_(0.0, 1.0)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, ):
        # Retained for compatibility with historical calling code.
        del epoch, y_obs

        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape

        if seq_len != 1:
            raise ValueError("MI-Only loss-output-gate model expects seq_length == 1.")

        if n_features < 2:
            raise ValueError("Model requires precipitation and PET.")

        hidden_size = self.hidden_size

        storage = x.new_zeros(1, hidden_size, )

        lagged_storage_state = x.new_zeros(1, hidden_size, )

        # Pre-update states and fluxes.
        discharge = x.new_zeros(batch_size, hidden_size, )

        storage_series = x.new_zeros(batch_size, hidden_size, )

        loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        loss_constrained = x.new_zeros(batch_size, hidden_size, )

        bypass = x.new_zeros(batch_size, hidden_size, )

        # Gate diagnostics.
        gate_input = x.new_zeros(batch_size, hidden_size, )

        gate_output = x.new_zeros(batch_size, hidden_size, )

        gate_loss_unconstrained = x.new_zeros(batch_size, hidden_size, )

        gate_loss_constrained = x.new_zeros(batch_size, hidden_size, )

        gate_remember = x.new_zeros(batch_size, hidden_size, )

        pet_mean = 2.9086
        pet_std = 1.8980

        # Base mass-partition scaling.
        exp_output = torch.exp(self.weight_r_yom)

        exp_loss = torch.exp(self.weight_r_ylm)

        exp_remember = torch.exp(self.weight_r_yfm)

        gate_sum = (exp_output + exp_loss + exp_remember)

        output_scale = (exp_output / gate_sum)

        loss_scale = (exp_loss / gate_sum)

        output_bias = (self.bias_b0_yom.unsqueeze(0))

        loss_bias = (self.bias_b0_ylm.unsqueeze(0))

        for b in range(batch_size):

            if b >= time_lag:
                lagged_storage = (lagged_storage_state)
            else:
                lagged_storage = (x.new_zeros(1, hidden_size, ))

            precipitation = (x[0, b, 0].reshape(1, 1))

            pet = (x[0, b, 1].reshape(1, 1))

            # ---------------------------------------------
            # Direct current + lagged storage output gate
            # ---------------------------------------------
            current_output_term = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom, )

            lagged_output_term = torch.mm((lagged_storage - c_mean) / c_std, self.weight_b2_yom, )

            output_state = (current_output_term + lagged_output_term)

            output_fraction = (output_scale * torch.sigmoid(output_state))

            # ---------------------------------------------
            # Direct storage + PET loss gate
            # ---------------------------------------------
            storage_loss_term = torch.addmm(loss_bias, (storage - c_mean) / c_std, self.weight_b1_ylm, )

            pet_loss_term = torch.mm((pet - pet_mean) / pet_std, self.weight_b2_ylm, )

            loss_state = (storage_loss_term + pet_loss_term)

            loss_fraction = (loss_scale * torch.sigmoid(loss_state))

            # PET constraint.
            if storage.item() > 0:
                loss_fraction_constrained = (loss_fraction - torch.relu(loss_fraction - pet / storage))
            else:
                loss_fraction_constrained = (loss_fraction)

            # Remember gate from mass conservation.
            remember_fraction = (1.0 - output_fraction - loss_fraction_constrained)

            # Pre-update diagnostics.
            storage_series[b, :] = (storage[0])

            discharge[b, :] = (output_fraction * storage)[0]

            loss_unconstrained[b, :] = (loss_fraction * storage)[0]

            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]

            gate_output[b, :] = (output_fraction[0])

            gate_loss_unconstrained[b, :] = (loss_fraction[0])

            gate_loss_constrained[b, :] = (loss_fraction_constrained[0])

            gate_remember[b, :] = (remember_fraction[0])

            current_preupdate_storage = (storage)

            # Input gate/bypass = 0.
            storage = (remember_fraction * storage + precipitation)

            if b >= time_lag:
                lagged_storage_state = (current_preupdate_storage)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
        )

class MCPBRNN_Generic_LossANNGate_PETconstraint(nn.Module):
    """Single-node MCP with a multi-dimensional PET-driven loss-gate ANN.

    The output gate depends on normalized storage. The loss gate uses a
    SELU hidden layer of dimension ``gate_dim_l`` driven by normalized PET.
    The PET-constrained loss gate is used to define the remember gate:

        gR = 1 - gO - gLc

    Parameter names and tensor shapes are preserved for old checkpoints.
    """

    def __init__(self, input_size, gate_dim_o, gate_dim_l, spinLen, traintimeLen,
                 batch_first=True, hidden_size=1, initial_forget_bias=0):
        super().__init__()
        del input_size, spinLen, traintimeLen, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("LossANNGate requires hidden_size == 1.")
        if gate_dim_o != 1:
            raise ValueError("LossANNGateOnly requires gate_dim_o == 1.")
        if gate_dim_l < 1:
            raise ValueError("gate_dim_l must be at least 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim_o = gate_dim_o
        self.gate_dim_l = gate_dim_l
        self.relu_l = nn.SELU()

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))
        self.bias_ln_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.relu_bias_l = nn.Parameter(torch.empty(1, gate_dim_l))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight_r_yom.uniform_(0, 1)
            self.weight_r_ylm.uniform_(0, 1)
            self.weight_r_yfm.uniform_(0, 1)
            self.bias_b0_yom.uniform_(0, 1)
            self.weight_b1_yom.uniform_(0, 1)
            self.weight_b2_ylm.uniform_(0, 1)
            self.bias_ln_ylm.uniform_(0, 1)
            self.relu_bias_l.uniform_(0, 1)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std):
        del epoch, y_obs
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, n_features = x.shape
        if seq_len != 1:
            raise ValueError("LossANNGate expects seq_length == 1.")
        if n_features < 2:
            raise ValueError("LossANNGate requires precipitation and PET.")

        storage = x.new_zeros(1, self.hidden_size)
        shape = (batch_size, self.hidden_size)
        discharge = x.new_zeros(*shape)
        storage_series = x.new_zeros(*shape)
        loss_unconstrained = x.new_zeros(*shape)
        loss_constrained = x.new_zeros(*shape)
        bypass = x.new_zeros(*shape)
        gate_input = x.new_zeros(*shape)
        gate_output = x.new_zeros(*shape)
        gate_loss_unconstrained = x.new_zeros(*shape)
        gate_loss_constrained = x.new_zeros(*shape)
        gate_remember = x.new_zeros(*shape)

        exp_o = torch.exp(self.weight_r_yom)
        exp_l = torch.exp(self.weight_r_ylm)
        exp_r = torch.exp(self.weight_r_yfm)
        gate_sum = exp_o + exp_l + exp_r
        output_scale, loss_scale = exp_o / gate_sum, exp_l / gate_sum
        output_bias = self.bias_b0_yom.unsqueeze(0)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            output_state = torch.addmm(output_bias, (storage - c_mean) / c_std, self.weight_b1_yom)
            output_fraction = output_scale * torch.sigmoid(output_state)

            pet_hidden = self.relu_l((pet.expand(1, self.gate_dim_l) - 2.9086) / 1.8980 - self.relu_bias_l)
            loss_state = torch.addmm(self.bias_ln_ylm, pet_hidden, self.weight_b2_ylm)
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            if storage.item() > 0:
                loss_fraction_constrained = loss_fraction - torch.relu(loss_fraction - pet / storage)
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = 1.0 - output_fraction - loss_fraction_constrained

            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]
            gate_output[b, :] = output_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]

            storage = remember_fraction * storage + precipitation

        return (
            discharge, storage_series, loss_unconstrained, loss_constrained, bypass,
            gate_input, gate_output, gate_loss_unconstrained,
            gate_loss_constrained, gate_remember,
        )

class MCPBRNN_Generic_OutputANNGate_PETconstraint(nn.Module):
    """Single-node MCP with a multi-dimensional storage-driven output-gate ANN.

    The output gate uses a SELU hidden layer of dimension ``gate_dim_o``
    driven by normalized storage. The loss gate remains a one-dimensional
    PET-dependent sigmoid gate. The remember gate is

        gR = 1 - gO - gLc

    where gLc is the PET-constrained loss gate.

    Parameter names and tensor shapes are preserved for old checkpoints.
    """

    def __init__(self, input_size, gate_dim_o, gate_dim_l, spinLen, traintimeLen,
                 batch_first=True, hidden_size=1, initial_forget_bias=0):
        super().__init__()
        del input_size, spinLen, traintimeLen, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("OutputANNGate requires hidden_size == 1.")
        if gate_dim_o < 1:
            raise ValueError("gate_dim_o must be at least 1.")
        if gate_dim_l != 1:
            raise ValueError("OutputANNGateOnly requires gate_dim_l == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim_o = gate_dim_o
        self.gate_dim_l = gate_dim_l
        self.relu_o = nn.SELU()

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))
        self.bias_ln_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.relu_bias_o = nn.Parameter(torch.empty(1, gate_dim_o))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight_r_yom.uniform_(0, 1)
            self.weight_r_ylm.uniform_(0, 1)
            self.weight_r_yfm.uniform_(0, 1)
            self.weight_b1_yom.uniform_(0, 1)
            self.bias_b0_ylm.uniform_(0, 1)
            self.weight_b2_ylm.uniform_(0, 1)
            self.bias_ln_yom.uniform_(0, 1)
            self.relu_bias_o.uniform_(0, 1)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std):
        del epoch, y_obs
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, n_features = x.shape
        if seq_len != 1:
            raise ValueError("OutputANNGate expects seq_length == 1.")
        if n_features < 2:
            raise ValueError("OutputANNGate requires precipitation and PET.")

        storage = x.new_zeros(1, self.hidden_size)
        shape = (batch_size, self.hidden_size)
        discharge = x.new_zeros(*shape)
        storage_series = x.new_zeros(*shape)
        loss_unconstrained = x.new_zeros(*shape)
        loss_constrained = x.new_zeros(*shape)
        bypass = x.new_zeros(*shape)
        gate_input = x.new_zeros(*shape)
        gate_output = x.new_zeros(*shape)
        gate_loss_unconstrained = x.new_zeros(*shape)
        gate_loss_constrained = x.new_zeros(*shape)
        gate_remember = x.new_zeros(*shape)

        exp_o = torch.exp(self.weight_r_yom)
        exp_l = torch.exp(self.weight_r_ylm)
        exp_r = torch.exp(self.weight_r_yfm)
        gate_sum = exp_o + exp_l + exp_r
        output_scale, loss_scale = exp_o / gate_sum, exp_l / gate_sum
        loss_bias = self.bias_b0_ylm.unsqueeze(0)

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            storage_hidden = self.relu_o(
                (storage.expand(1, self.gate_dim_o) - c_mean) / c_std - self.relu_bias_o
            )
            output_state = torch.addmm(self.bias_ln_yom, storage_hidden, self.weight_b1_yom)
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_state = torch.addmm(loss_bias, (pet - 2.9086) / 1.8980, self.weight_b2_ylm)
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            if storage.item() > 0:
                loss_fraction_constrained = loss_fraction - torch.relu(loss_fraction - pet / storage)
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = 1.0 - output_fraction - loss_fraction_constrained

            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]
            gate_output[b, :] = output_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]

            storage = remember_fraction * storage + precipitation

        return (
            discharge, storage_series, loss_unconstrained, loss_constrained, bypass,
            gate_input, gate_output, gate_loss_unconstrained,
            gate_loss_constrained, gate_remember,
        )

class MCPBRNN_Generic_ANNGate_PETconstraint_Generic(nn.Module):
    """Single-node MCP with ANN output and loss gates.

    Output gate: normalized storage -> shifted SELU hidden layer -> sigmoid.
    Loss gate: normalized PET -> shifted SELU hidden layer -> sigmoid -> PET constraint.

    The remember gate is gR = 1 - gO - gLc. Parameter names and tensor
    shapes are preserved for compatibility with historical checkpoints.
    """

    def __init__(self, input_size, gate_dim_o, gate_dim_l, spinLen, traintimeLen,
                 batch_first=True, hidden_size=1, initial_forget_bias=0):
        super().__init__()
        del input_size, spinLen, traintimeLen, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("ANNGate model requires hidden_size == 1.")
        if gate_dim_o < 1 or gate_dim_l < 1:
            raise ValueError("gate dimensions must be >= 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim_o = gate_dim_o
        self.gate_dim_l = gate_dim_l
        self.relu_o = nn.SELU()
        self.relu_l = nn.SELU()

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(gate_dim_o, hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(gate_dim_l, hidden_size))
        self.bias_ln_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ln_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.relu_bias_o = nn.Parameter(torch.empty(1, gate_dim_o))
        self.relu_bias_l = nn.Parameter(torch.empty(1, gate_dim_l))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight_r_yom.uniform_(0, 1)
            self.weight_r_ylm.uniform_(0, 1)
            self.weight_r_yfm.uniform_(0, 1)
            self.weight_b1_yom.uniform_(0, 1)
            self.weight_b2_ylm.uniform_(0, 1)
            self.bias_ln_yom.uniform_(0, 1)
            self.bias_ln_ylm.uniform_(0, 1)
            self.relu_bias_o.uniform_(0, 1)
            self.relu_bias_l.uniform_(0, 1)

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std):
        del epoch, y_obs
        if c_std == 0:
            raise ValueError("c_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, n_features = x.shape
        if seq_len != 1:
            raise ValueError("ANNGate model expects seq_length == 1.")
        if n_features < 2:
            raise ValueError("ANNGate model requires precipitation and PET.")

        storage = x.new_zeros(1, self.hidden_size)
        shape = (batch_size, self.hidden_size)
        discharge = x.new_zeros(*shape)
        storage_series = x.new_zeros(*shape)
        loss_unconstrained = x.new_zeros(*shape)
        loss_constrained = x.new_zeros(*shape)
        bypass = x.new_zeros(*shape)
        gate_input = x.new_zeros(*shape)
        gate_output = x.new_zeros(*shape)
        gate_loss_unconstrained = x.new_zeros(*shape)
        gate_loss_constrained = x.new_zeros(*shape)
        gate_remember = x.new_zeros(*shape)

        exp_o = torch.exp(self.weight_r_yom)
        exp_l = torch.exp(self.weight_r_ylm)
        exp_r = torch.exp(self.weight_r_yfm)
        gate_sum = exp_o + exp_l + exp_r
        output_scale, loss_scale = exp_o / gate_sum, exp_l / gate_sum

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            output_hidden = self.relu_o(
                (storage.expand(1, self.gate_dim_o) - c_mean) / c_std - self.relu_bias_o
            )
            output_state = torch.addmm(self.bias_ln_yom, output_hidden, self.weight_b1_yom)
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_hidden = self.relu_l(
                (pet.expand(1, self.gate_dim_l) - 2.9086) / 1.8980 - self.relu_bias_l
            )
            loss_state = torch.addmm(self.bias_ln_ylm, loss_hidden, self.weight_b2_ylm)
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            if storage.item() > 0:
                loss_fraction_constrained = loss_fraction - torch.relu(loss_fraction - pet / storage)
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = 1.0 - output_fraction - loss_fraction_constrained

            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]
            gate_output[b, :] = output_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]

            storage = remember_fraction * storage + precipitation

        return (
            discharge, storage_series, loss_unconstrained, loss_constrained, bypass,
            gate_input, gate_output, gate_loss_unconstrained,
            gate_loss_constrained, gate_remember,
        )

class MCPBRNN_PETconstraint_MassRelax_Regular(nn.Module):
    """Single-node MCP with regular mass relaxation.

    The hydrologic output and PET-constrained loss gates follow the M5 core.
    An additional signed mass-relaxation flux drives storage toward an
    equilibrium level controlled by ``bias_b0_yrm`` and ``scale_mr = 500``.

    The remember gate remains:
        gR = 1 - gO - gLc

    Mass relaxation is applied separately in the state update:
        S[t+1] = gR*S[t] + P[t] - MR_flux

    Parameter names and shapes are preserved for historical checkpoints.
    """

    def __init__(self, input_size, gate_dim, gate_dim_ucorr, spinLen, traintimeLen,
                 batch_first=True, hidden_size=1, initial_forget_bias=0):
        super().__init__()
        del input_size, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MassRelax_Regular requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim = gate_dim
        self.gate_dim_ucorr = gate_dim_ucorr
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight_r_yom.uniform_(0, 1)
            self.weight_r_ylm.uniform_(0, 1)
            self.weight_r_yfm.uniform_(0, 1)
            self.weight_r_yvm.uniform_(0, 1)
            self.bias_b0_yom.uniform_(0, 1)
            self.weight_b1_yom.uniform_(0, 1)
            self.bias_b0_ylm.uniform_(0, 1)
            self.weight_b2_ylm.uniform_(0, 1)
            self.weight_s_yvm.uniform_(0, 1)
            self.bias_b0_yrm.uniform_(0, 1)

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        del epoch
        if p_std == 0:
            raise ValueError("p_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape
        if seq_len != 1:
            raise ValueError("MassRelax_Regular expects seq_length == 1.")
        if n_features < 2:
            raise ValueError("MassRelax_Regular requires precipitation and PET.")

        storage = x.new_zeros(1, self.hidden_size)
        shape = (batch_size, self.hidden_size)

        discharge = x.new_zeros(*shape)
        storage_series = x.new_zeros(*shape)
        loss_unconstrained = x.new_zeros(*shape)
        loss_constrained = x.new_zeros(*shape)
        bypass = x.new_zeros(*shape)
        gate_input = x.new_zeros(*shape)
        gate_output = x.new_zeros(*shape)
        gate_loss_unconstrained = x.new_zeros(*shape)
        gate_loss_constrained = x.new_zeros(*shape)
        gate_remember = x.new_zeros(*shape)
        gate_mass_relaxation = x.new_zeros(*shape)
        mass_relaxation_flux = x.new_zeros(*shape)
        obs_std = x.new_zeros(*shape)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])

        exp_o = torch.exp(self.weight_r_yom)
        exp_l = torch.exp(self.weight_r_ylm)
        exp_r = torch.exp(self.weight_r_yfm)
        gate_sum = exp_o + exp_l + exp_r

        output_scale = exp_o / gate_sum
        loss_scale = exp_l / gate_sum
        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)
        mr_bias = self.bias_b0_yrm.unsqueeze(0)
        scale_mr = 500.0

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            output_state = torch.addmm(output_bias, (storage - p_mean) / p_std, self.weight_b1_yom)
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_state = torch.addmm(loss_bias, (pet - 2.9086) / 1.8980, self.weight_b2_ylm)
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            if storage.item() > 0:
                loss_fraction_constrained = loss_fraction - torch.relu(loss_fraction - pet / storage)
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = 1.0 - output_fraction - loss_fraction_constrained

            mr_state = torch.mm(storage / scale_mr - torch.exp(mr_bias), torch.exp(self.weight_s_yvm))
            mr_raw = torch.sigmoid(self.weight_r_yvm) * torch.tanh(mr_state)
            mr_gate = mr_raw - torch.relu(mr_raw - remember_fraction)
            mr_flux = mr_gate * torch.abs(storage - torch.exp(mr_bias) * scale_mr)

            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]
            gate_output[b, :] = output_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]
            gate_mass_relaxation[b, :] = mr_gate[0]
            mass_relaxation_flux[b, :] = mr_flux[0]

            storage = remember_fraction * storage + precipitation - mr_flux
            obs_std[b, :] = obsstd

        h_nout = torch.cat((discharge, obs_std), dim=1)

        return (
            discharge, storage_series, loss_unconstrained, loss_constrained, bypass,
            gate_input, gate_output, gate_loss_unconstrained, gate_loss_constrained,
            gate_remember, h_nout, obs_std, gate_mass_relaxation, mass_relaxation_flux,
        )

class MCPBRNN_PETconstraint_MassRelax_Indepedent(nn.Module):
    """Single-node MCP with independent-sign mass relaxation.

    The M5 output/loss/remember core is unchanged. Mass relaxation uses only
    the sign of the storage departure from an equilibrium level:

        sign(S / 500 - exp(bias_b0_yrm))

    Its magnitude is controlled by sigmoid(weight_r_yvm), capped above by
    the remember gate, and applied as a signed state correction.

    The historical misspelling ``Indepedent`` is preserved for checkpoint
    and script compatibility.
    """

    def __init__(self, input_size, gate_dim, gate_dim_ucorr, spinLen, traintimeLen,
                 batch_first=True, hidden_size=1, initial_forget_bias=0):
        super().__init__()
        del input_size, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MassRelax_Indepedent requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim = gate_dim
        self.gate_dim_ucorr = gate_dim_ucorr
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight_r_yom.uniform_(0, 1)
            self.weight_r_ylm.uniform_(0, 1)
            self.weight_r_yfm.uniform_(0, 1)
            self.weight_r_yvm.uniform_(0, 1)
            self.bias_b0_yom.uniform_(0, 1)
            self.weight_b1_yom.uniform_(0, 1)
            self.bias_b0_ylm.uniform_(0, 1)
            self.weight_b2_ylm.uniform_(0, 1)
            self.bias_b0_yrm.uniform_(0, 1)

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        del epoch

        if p_std == 0:
            raise ValueError("p_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape
        if seq_len != 1:
            raise ValueError("MassRelax_Indepedent expects seq_length == 1.")
        if n_features < 2:
            raise ValueError("MassRelax_Indepedent requires precipitation and PET.")

        storage = x.new_zeros(1, self.hidden_size)
        shape = (batch_size, self.hidden_size)

        discharge = x.new_zeros(*shape)
        storage_series = x.new_zeros(*shape)
        loss_unconstrained = x.new_zeros(*shape)
        loss_constrained = x.new_zeros(*shape)
        bypass = x.new_zeros(*shape)
        gate_input = x.new_zeros(*shape)
        gate_output = x.new_zeros(*shape)
        gate_loss_unconstrained = x.new_zeros(*shape)
        gate_loss_constrained = x.new_zeros(*shape)
        gate_remember = x.new_zeros(*shape)
        gate_mass_relaxation = x.new_zeros(*shape)
        obs_std = x.new_zeros(*shape)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])

        exp_o = torch.exp(self.weight_r_yom)
        exp_l = torch.exp(self.weight_r_ylm)
        exp_r = torch.exp(self.weight_r_yfm)
        gate_sum = exp_o + exp_l + exp_r

        output_scale = exp_o / gate_sum
        loss_scale = exp_l / gate_sum
        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)
        mr_bias = self.bias_b0_yrm.unsqueeze(0)
        scale_mr = 500.0

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            output_state = torch.addmm(output_bias, (storage - p_mean) / p_std, self.weight_b1_yom)
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_state = torch.addmm(loss_bias, (pet - 2.9086) / 1.8980, self.weight_b2_ylm)
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            if storage.item() > 0:
                loss_fraction_constrained = loss_fraction - torch.relu(loss_fraction - pet / storage)
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = 1.0 - output_fraction - loss_fraction_constrained

            mr_sign = torch.sign(storage / scale_mr - torch.exp(mr_bias))
            mr_raw = torch.sigmoid(self.weight_r_yvm) * mr_sign
            mr_gate = mr_raw - torch.relu(mr_raw - remember_fraction)
            mr_flux = mr_gate * torch.abs(storage - torch.exp(mr_bias) * scale_mr)

            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]
            gate_output[b, :] = output_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]
            gate_mass_relaxation[b, :] = mr_gate[0]

            storage = remember_fraction * storage + precipitation - mr_flux
            obs_std[b, :] = obsstd

        h_nout = torch.cat((discharge, obs_std), dim=1)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            h_nout,
            obs_std,
            gate_mass_relaxation,
        )

class MCPBRNN_PETconstraint_MassRelax_Regular_Relaxed(nn.Module):
    """Single-node MCP with the relaxed regular mass-relaxation formulation.

    This is distinct from ``MCPBRNN_PETconstraint_MassRelax_Regular``:
    the equilibrium level uses ``bias_b0_yrm`` directly rather than
    ``exp(bias_b0_yrm)``.

    The remember gate remains gR = 1 - gO - gLc, while the MR correction is
    applied separately in the storage update. Parameter names and shapes are
    preserved so historical checkpoints still strict-load.
    """

    def __init__(self, input_size, gate_dim, gate_dim_ucorr, spinLen, traintimeLen,
                 batch_first=True, hidden_size=1, initial_forget_bias=0):
        super().__init__()
        del input_size, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MassRelax_Regular_Relaxed requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim = gate_dim
        self.gate_dim_ucorr = gate_dim_ucorr
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight_r_yom.uniform_(0, 1)
            self.weight_r_ylm.uniform_(0, 1)
            self.weight_r_yfm.uniform_(0, 1)
            self.weight_r_yvm.uniform_(0, 1)
            self.bias_b0_yom.uniform_(0, 1)
            self.weight_b1_yom.uniform_(0, 1)
            self.bias_b0_ylm.uniform_(0, 1)
            self.weight_b2_ylm.uniform_(0, 1)
            self.weight_s_yvm.uniform_(0, 1)
            self.bias_b0_yrm.uniform_(0, 1)

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        del epoch

        if p_std == 0:
            raise ValueError("p_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape
        if seq_len != 1:
            raise ValueError("MassRelax_Regular_Relaxed expects seq_length == 1.")
        if n_features < 2:
            raise ValueError("MassRelax_Regular_Relaxed requires precipitation and PET.")

        storage = x.new_zeros(1, self.hidden_size)
        shape = (batch_size, self.hidden_size)

        discharge = x.new_zeros(*shape)
        storage_series = x.new_zeros(*shape)
        loss_unconstrained = x.new_zeros(*shape)
        loss_constrained = x.new_zeros(*shape)
        bypass = x.new_zeros(*shape)
        gate_input = x.new_zeros(*shape)
        gate_output = x.new_zeros(*shape)
        gate_loss_unconstrained = x.new_zeros(*shape)
        gate_loss_constrained = x.new_zeros(*shape)
        gate_remember = x.new_zeros(*shape)
        gate_mass_relaxation = x.new_zeros(*shape)
        obs_std = x.new_zeros(*shape)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])

        exp_o = torch.exp(self.weight_r_yom)
        exp_l = torch.exp(self.weight_r_ylm)
        exp_r = torch.exp(self.weight_r_yfm)
        gate_sum = exp_o + exp_l + exp_r

        output_scale = exp_o / gate_sum
        loss_scale = exp_l / gate_sum
        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)
        mr_bias = self.bias_b0_yrm.unsqueeze(0)
        scale_mr = 500.0

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            output_state = torch.addmm(output_bias, (storage - p_mean) / p_std, self.weight_b1_yom)
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_state = torch.addmm(loss_bias, (pet - 2.9086) / 1.8980, self.weight_b2_ylm)
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            if storage.item() > 0:
                loss_fraction_constrained = loss_fraction - torch.relu(loss_fraction - pet / storage)
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = 1.0 - output_fraction - loss_fraction_constrained

            mr_state = torch.mm(storage / scale_mr - mr_bias, torch.exp(self.weight_s_yvm))
            mr_raw = torch.sigmoid(self.weight_r_yvm) * torch.tanh(mr_state)
            mr_gate = mr_raw - torch.relu(mr_raw - remember_fraction)
            mr_flux = mr_gate * torch.abs(storage - mr_bias * scale_mr)

            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]
            gate_output[b, :] = output_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]
            gate_mass_relaxation[b, :] = mr_gate[0]

            storage = remember_fraction * storage + precipitation - mr_flux
            obs_std[b, :] = obsstd

        h_nout = torch.cat((discharge, obs_std), dim=1)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            h_nout,
            obs_std,
            gate_mass_relaxation,
        )

class MCPBRNN_PETconstraint_MassRelax_Independent_Relaxed(nn.Module):
    """Single-node MCP with independent-sign relaxed mass relaxation.

    This is the relaxed counterpart of the historical independent MR case.
    The equilibrium level uses ``bias_b0_yrm`` directly rather than
    ``exp(bias_b0_yrm)``.

    The remember gate remains gR = 1 - gO - gLc. Mass relaxation is applied
    separately in the state update. Parameter names and shapes are preserved
    so historical checkpoints strict-load unchanged.
    """

    def __init__(self, input_size, gate_dim, gate_dim_ucorr, spinLen, traintimeLen,
                 batch_first=True, hidden_size=1, initial_forget_bias=0):
        super().__init__()
        del input_size, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MassRelax_Independent_Relaxed requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gate_dim = gate_dim
        self.gate_dim_ucorr = gate_dim_ucorr
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen

        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight_r_yom.uniform_(0, 1)
            self.weight_r_ylm.uniform_(0, 1)
            self.weight_r_yfm.uniform_(0, 1)
            self.weight_r_yvm.uniform_(0, 1)
            self.bias_b0_yom.uniform_(0, 1)
            self.weight_b1_yom.uniform_(0, 1)
            self.bias_b0_ylm.uniform_(0, 1)
            self.weight_b2_ylm.uniform_(0, 1)
            self.bias_b0_yrm.uniform_(0, 1)

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
        del epoch

        if p_std == 0:
            raise ValueError("p_std must be non-zero.")

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, n_features = x.shape
        if seq_len != 1:
            raise ValueError("MassRelax_Independent_Relaxed expects seq_length == 1.")
        if n_features < 2:
            raise ValueError("MassRelax_Independent_Relaxed requires precipitation and PET.")

        storage = x.new_zeros(1, self.hidden_size)
        shape = (batch_size, self.hidden_size)

        discharge = x.new_zeros(*shape)
        storage_series = x.new_zeros(*shape)
        loss_unconstrained = x.new_zeros(*shape)
        loss_constrained = x.new_zeros(*shape)
        bypass = x.new_zeros(*shape)
        gate_input = x.new_zeros(*shape)
        gate_output = x.new_zeros(*shape)
        gate_loss_unconstrained = x.new_zeros(*shape)
        gate_loss_constrained = x.new_zeros(*shape)
        gate_remember = x.new_zeros(*shape)
        gate_mass_relaxation = x.new_zeros(*shape)
        obs_std = x.new_zeros(*shape)

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])

        exp_o = torch.exp(self.weight_r_yom)
        exp_l = torch.exp(self.weight_r_ylm)
        exp_r = torch.exp(self.weight_r_yfm)
        gate_sum = exp_o + exp_l + exp_r

        output_scale = exp_o / gate_sum
        loss_scale = exp_l / gate_sum
        output_bias = self.bias_b0_yom.unsqueeze(0)
        loss_bias = self.bias_b0_ylm.unsqueeze(0)
        mr_bias = self.bias_b0_yrm.unsqueeze(0)
        scale_mr = 500.0

        for b in range(time_lag, batch_size):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            output_state = torch.addmm(output_bias, (storage - p_mean) / p_std, self.weight_b1_yom)
            output_fraction = output_scale * torch.sigmoid(output_state)

            loss_state = torch.addmm(loss_bias, (pet - 2.9086) / 1.8980, self.weight_b2_ylm)
            loss_fraction = loss_scale * torch.sigmoid(loss_state)

            if storage.item() > 0:
                loss_fraction_constrained = loss_fraction - torch.relu(loss_fraction - pet / storage)
            else:
                loss_fraction_constrained = loss_fraction

            remember_fraction = 1.0 - output_fraction - loss_fraction_constrained

            mr_sign = torch.sign(storage / scale_mr - mr_bias)
            mr_raw = torch.sigmoid(self.weight_r_yvm) * mr_sign
            mr_gate = mr_raw - torch.relu(mr_raw - remember_fraction)
            mr_flux = mr_gate * torch.abs(storage - mr_bias * scale_mr)

            storage_series[b, :] = storage[0]
            discharge[b, :] = (output_fraction * storage)[0]
            loss_unconstrained[b, :] = (loss_fraction * storage)[0]
            loss_constrained[b, :] = (loss_fraction_constrained * storage)[0]
            gate_output[b, :] = output_fraction[0]
            gate_loss_unconstrained[b, :] = loss_fraction[0]
            gate_loss_constrained[b, :] = loss_fraction_constrained[0]
            gate_remember[b, :] = remember_fraction[0]
            gate_mass_relaxation[b, :] = mr_gate[0]

            storage = remember_fraction * storage + precipitation - mr_flux
            obs_std[b, :] = obsstd

        h_nout = torch.cat((discharge, obs_std), dim=1)

        return (
            discharge,
            storage_series,
            loss_unconstrained,
            loss_constrained,
            bypass,
            gate_input,
            gate_output,
            gate_loss_unconstrained,
            gate_loss_constrained,
            gate_remember,
            h_nout,
            obs_std,
            gate_mass_relaxation,
        )
