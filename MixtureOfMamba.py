"""
# Copyright (c) 2023, Tri Dao, Albert Gu.
# Copyright (c) 2024, Modifications by Weixin Liang.

This is the core implementation of Mixture-of-Mamba as described in our paper.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from torch import Tensor

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn, RMSNorm
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# Please implement your own configuration class with the required attributes. 
from src.args import ModelArgs


# # Weixin: same as transformer.py
# def get_modality_mask(modality_src_tokens: torch.Tensor, n_modalities: int):
#     modality_masks = modality_src_tokens.unsqueeze(0) == torch.arange(
#         n_modalities, device=modality_src_tokens.device
#     ).unsqueeze(1)
#     modality_masks[:, -1] = True
#     return modality_masks


class MixtureOfMamba(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.args = args

        self.n_modalities = args.n_modalities

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        ### Weixin:
        # # 1. **Input Projection Layer (`self.in_proj`)**:
        # # - Shape: `(d_model, d_inner * 2)` with `d_inner = expand * d_model`
        # # - Learnable weights: `(d_model, d_model * expand * 2) = (1024, 2048)`, which is \(1024 \times 2048 = 2,097,152\).
        # # - Bias: `(d_inner * 2) = 2048` learnable biases.
        # # - **Total for `in_proj`:** \(2,097,152 + 2048 = 2,099,200\)

        if getattr(self.args, "do_not_split_in_proj", False):
            self.in_proj = nn.Linear(
                self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
            )
        else:
            ### Weixin: This is the new equivalent of in_proj.
            expert_list = [
                nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
                for _ in range(self.n_modalities)
            ]
            self.local_experts_in_proj = torch.nn.ModuleList(expert_list)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # # Weixin - Let's say this is the primary target for Sparsity

        if getattr(self.args, "do_not_split_x_proj", False):

            self.x_proj = nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )  # Weixin - This is to be duplicated.

        else:

            ### Weixin: This is the new equivalent of wq.
            expert_list = [
                nn.Linear(
                    self.d_inner,
                    self.dt_rank + self.d_state * 2,
                    bias=False,
                    **factory_kwargs,
                )
                for _ in range(self.n_modalities)
            ]
            self.local_experts_x_proj = torch.nn.ModuleList(expert_list)

        if getattr(self.args, "do_not_split_dt_proj", False):

            self.dt_proj = nn.Linear(
                self.dt_rank, self.d_inner, bias=True, **factory_kwargs
            )

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.dt_proj.bias._no_reinit = True

        else:

            ### Weixin: This is the new equivalent of wq.
            expert_list = [
                nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
                for _ in range(self.n_modalities)
            ]
            self.local_experts_dt_proj = torch.nn.ModuleList(expert_list)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                for modality_idx in range(self.n_modalities):
                    nn.init.constant_(
                        self.local_experts_dt_proj[modality_idx].weight, dt_init_std
                    )
            elif dt_init == "random":
                for modality_idx in range(self.n_modalities):
                    nn.init.uniform_(
                        self.local_experts_dt_proj[modality_idx].weight,
                        -dt_init_std,
                        dt_init_std,
                    )
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            for modality_idx in range(self.n_modalities):
                dt = torch.exp(
                    torch.rand(self.d_inner, **factory_kwargs)
                    * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                ).clamp(min=dt_init_floor)
                # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                with torch.no_grad():
                    self.local_experts_dt_proj[modality_idx].bias.copy_(inv_dt)
                # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
                self.local_experts_dt_proj[modality_idx].bias._no_reinit = True

            # Weixin: end init self.local_experts_dt_proj

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        ### Weixin:
        # # 6. **Output Projection Layer (`self.out_proj`)**:
        # #    - Shape: `(d_inner, d_model)` or `(2048, 1024)`, totaling \(2048 \times 1024 = 2,097,152\).
        # #    - **Total for `out_proj`:** 2,097,152

        if getattr(self.args, "do_not_split_out_proj", False):
            self.out_proj = nn.Linear(
                self.d_inner, self.d_model, bias=bias, **factory_kwargs
            )
        else:
            ### Weixin: This is the new equivalent of out_proj.
            expert_list = [
                nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
                for _ in range(self.n_modalities)
            ]
            self.local_experts_out_proj = torch.nn.ModuleList(expert_list)

    def forward(
        self,
        hidden_states,
        modality_masks = None, 
        inference_params=None,
    ):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out


        if getattr(self.args, "do_not_split_in_proj", False):
            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        else:

            # Initial reshaping for projection
            x = rearrange(hidden_states, "b l d -> (b l) d")  # (b*l) d
            expert_outputs = []
            # Apply each modality-specific projection layer on its respective input subset
            for i in range(self.n_modalities):
                expert = self.local_experts_in_proj[i]
                expert_input = x[modality_masks[i]]

                expert_output = expert(
                    expert_input,
                )  # Result shape: (num_modality_i, d_inner * 2)
                expert_outputs.append(expert_output)

            # Merging the outputs from different modality-specific layers
            merged_output = torch.empty(
                (x.size(0), expert_outputs[0].size(1)),
                device=x.device,
                dtype=x.dtype,
            )
            for i in range(self.n_modalities - 1, -1, -1):
                expert_output = expert_outputs[i]
                with torch.profiler.record_function(
                    f"assembling modality expert outputs: {i} @ {expert_output.shape}"
                ):
                    merged_output[modality_masks[i]] = expert_output

            # Final reshaping to match the expected structure `(b, d_inner * 2, l)`
            xz = rearrange(
                merged_output, "(b l) d -> b d l", b=hidden_states.size(0), l=seqlen
            )
            # Weixin: end use self.local_experts_in_proj

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        if False:
            raise NotImplementedError("Not using this path")
            # if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states

            # out = mamba_inner_fn(
            #     xz,
            #     self.conv1d.weight,
            #     self.conv1d.bias,
            #     self.x_proj.weight,
            #     self.dt_proj.weight,
            #     self.out_proj.weight,
            #     self.out_proj.bias,
            #     A,
            #     None,  # input-dependent B
            #     None,  # input-dependent C
            #     self.D.float(),
            #     delta_bias=self.dt_proj.bias.float(),
            #     delta_softplus=True,
            # )
        else:

            x, z = xz.chunk(2, dim=1)

            if conv_state is not None:
                conv_state.copy_(
                    F.pad(x, (self.d_conv - x.shape[-1], 0))
                )  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )


            if getattr(self.args, "do_not_split_x_proj", False):
                x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            else:


                x_dbl = rearrange(x, "b d l -> (b l) d")  # (bl d)

                expert_outputs = []
                for i in range(self.n_modalities):
                    expert = self.local_experts_x_proj[i]
                    expert_input = x_dbl[modality_masks[i]]

                    expert_output = expert(
                        expert_input,
                    )
                    expert_outputs.append(expert_output)

                merged_output = torch.empty(
                    (x_dbl.size(0), expert_outputs[0].size(1)),
                    device=x_dbl.device,
                    dtype=x_dbl.dtype,
                )
                for i in range(self.n_modalities - 1, -1, -1):
                    expert_output = expert_outputs[i]
                    with torch.profiler.record_function(
                        f"assembling modality expert outputs: {i} @ {expert_output.shape}"
                    ):
                        merged_output[modality_masks[i]] = expert_output

                x_dbl = merged_output


            dt, B, C = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )

            if getattr(self.args, "do_not_split_dt_proj", False):
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            else:

                expert_outputs = []
                for i in range(self.n_modalities):
                    expert = self.local_experts_dt_proj[i]
                    expert_input = dt[
                        modality_masks[i]
                    ] 

                    expert_output = expert(
                        expert_input
                    )  
                    expert_outputs.append(expert_output)

                merged_dt_output = torch.empty(
                    (dt.size(0), expert_outputs[0].size(1)),
                    device=dt.device,
                    dtype=dt.dtype,
                )
                for i in range(self.n_modalities - 1, -1, -1):
                    expert_output = expert_outputs[i]
                    with torch.profiler.record_function(
                        f"assembling modality expert outputs: {i} @ {expert_output.shape}"
                    ):
                        merged_dt_output[modality_masks[i]] = expert_output

                dt = rearrange(merged_dt_output, "(b l) d -> b d l", l=seqlen)

            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)

            if getattr(self.args, "do_not_split_out_proj", False):
                y = rearrange(y, "b d l -> b l d")
                out = self.out_proj(y)
            else:

                # Weixin: start out_proj

                y = rearrange(
                    y, "b d l ->  (b l) d"
                )  # (bl d) ### Note that different from original

                expert_outputs = []
                for i in range(self.n_modalities):
                    expert = self.local_experts_out_proj[i]
                    expert_input = y[modality_masks[i]]

                    expert_output = expert(
                        expert_input,
                    )
                    expert_outputs.append(expert_output)

                merged_output = torch.empty(
                    (y.size(0), expert_outputs[0].size(1)),
                    device=y.device,
                    dtype=y.dtype,
                )
                for i in range(self.n_modalities - 1, -1, -1):
                    expert_output = expert_outputs[i]
                    with torch.profiler.record_function(
                        f"assembling modality expert outputs: {i} @ {expert_output.shape}"
                    ):
                        merged_output[modality_masks[i]] = expert_output

                out = merged_output

                out = rearrange(out, "(b l) d -> b l d", l=seqlen)


        return out



    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
