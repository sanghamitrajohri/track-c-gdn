"""B200-tuned Triton GDN decode kernel for gdn_decode_qk4_v8_d128_k_last."""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


HQ = 4
HV = 8
D = 128
BV = 16
NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def _softplus_20(x):
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.jit
def _gdn_decode_headloop_kernel(
    q,
    k,
    v,
    state,
    A_log,
    a,
    dt_bias,
    b,
    out,
    new_state,
    scale,
    HQ: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    batch = pid_bh // HV
    hv = pid_bh - batch * HV
    h = hv // 2

    offs_k = tl.arange(0, D)

    qk_base = (batch * HQ + h) * D
    v_base = (batch * HV + hv) * D
    state_base = (batch * HV + hv) * D * D
    gate_base = batch * HV + hv

    q_vec = tl.load(q + qk_base + offs_k).to(tl.float32) * scale
    k_vec = tl.load(k + qk_base + offs_k).to(tl.float32)

    a_val = tl.load(a + gate_base).to(tl.float32)
    b_val = tl.load(b + gate_base).to(tl.float32)
    log_g = -tl.exp(tl.load(A_log + hv).to(tl.float32)) * _softplus_20(
        a_val + tl.load(dt_bias + hv).to(tl.float32)
    )
    g_val = tl.exp(log_g)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    for start_v in tl.static_range(0, D, BV):
        offs_v = start_v + tl.arange(0, BV)
        v_vec = tl.load(v + v_base + offs_v).to(tl.float32)
        state_offsets = state_base + offs_v[None, :] * D + offs_k[:, None]

        h_mat = tl.load(state + state_offsets).to(tl.float32) * g_val
        old_v = tl.sum(h_mat * k_vec[:, None], axis=0)
        delta = beta * (v_vec - old_v)
        h_new = h_mat + k_vec[:, None] * delta[None, :]
        out_vec = tl.sum(h_new * q_vec[:, None], axis=0)

        tl.store(out + v_base + offs_v, out_vec.to(out.dtype.element_ty))
        tl.store(new_state + state_offsets, h_new)


def _prepare_state(
    q: torch.Tensor,
    state: Optional[torch.Tensor],
) -> torch.Tensor:
    if state is None:
        return torch.zeros(
            q.shape[0],
            HV,
            D,
            D,
            dtype=torch.float32,
            device=q.device,
        )
    if not state.is_contiguous():
        return state.contiguous()
    return state


@torch.no_grad()
def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float],
):
    if scale is None or float(scale) == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    else:
        scale = float(scale)

    assert q.shape[1:] == (1, HQ, D)
    assert k.shape[1:] == (1, HQ, D)
    assert v.shape[1:] == (1, HV, D)
    assert q.shape[0] == v.shape[0]

    state = _prepare_state(q, state)
    output = torch.empty(q.shape[0], 1, HV, D, dtype=q.dtype, device=q.device)
    new_state = torch.empty_like(state)

    _gdn_decode_headloop_kernel[(q.shape[0] * HV,)](
        q.squeeze(1),
        k.squeeze(1),
        v.squeeze(1),
        state,
        A_log.float(),
        a.squeeze(1),
        dt_bias.float(),
        b.squeeze(1),
        output,
        new_state,
        scale,
        HQ=HQ,
        HV=HV,
        D=D,
        BV=BV,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return output, new_state
