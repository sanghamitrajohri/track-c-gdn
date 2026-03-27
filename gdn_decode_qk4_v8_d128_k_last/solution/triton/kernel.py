# Gated Delta Net decode kernel for the MLSys 2026 FlashInfer contest.
# Definition: gdn_decode_qk4_v8_d128_k_last
#
# Decode is single-token (seq_len=1) per batch element.
# Inputs:  q,k: [B,1,HQ,D], v: [B,1,HV,D], state: [B,HV,D,D] (k-last layout)
# Outputs: output: [B,1,HV,D], new_state: [B,HV,D,D]

import math
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_SIZE = 128
HEAD_EXPANSION = NUM_V_HEADS // NUM_Q_HEADS  # 2


@triton.jit
def gdn_decode_kernel(
    q_ptr,       # [B, HV, D] float (already head-expanded)
    k_ptr,       # [B, HV, D] float
    v_ptr,       # [B, HV, D] float
    g_ptr,       # [B, HV] float
    beta_ptr,    # [B, HV] float
    state_ptr,   # [B, HV, D, D] float32, k-last layout [V, K]
    out_ptr,     # [B, HV, D] bf16
    new_state_ptr,  # [B, HV, D, D] float32
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Each program handles one (batch, head) pair.
    Tiles over V dimension of state [V, K] with block size BV.

    For each tile of BV rows (V-indices):
      1. Load state tile [BV, K] and apply gate: gated = g * state
      2. old_v[v] = dot(gated[v,:], k)  -- retrieve from state
      3. new_v[v] = beta * v[v] + (1-beta) * old_v[v]  -- delta rule blending
      4. new_state[v,:] = gated[v,:] + (new_v[v] - old_v[v]) * k  -- rank-1 update
      5. output[v] = scale * dot(q, new_state[v,:])
    """
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H

    # Load scalars: gate and beta
    gb_offset = i_b * H + i_h
    b_g = tl.load(g_ptr + gb_offset).to(tl.float32)
    b_beta = tl.load(beta_ptr + gb_offset).to(tl.float32)

    # Load q, k vectors: [D]
    qkv_base = (i_b * H + i_h) * D
    offs_d = tl.arange(0, D)  # [D] = [128]

    b_q = tl.load(q_ptr + qkv_base + offs_d).to(tl.float32)
    b_k = tl.load(k_ptr + qkv_base + offs_d).to(tl.float32)

    # State base offset for this (batch, head)
    state_base = (i_b * H + i_h) * D * D

    # Process V dimension in tiles of BV
    for i_v in range(0, D, BV):
        offs_v = i_v + tl.arange(0, BV)  # [BV]

        # Load v[offs_v] -- the value vector elements for this tile
        b_v_tile = tl.load(v_ptr + qkv_base + offs_v).to(tl.float32)  # [BV]

        # Load state[offs_v, :] -- [BV, D] tile of [V, K] state
        p_state = state_base + offs_v[:, None] * D + offs_d[None, :]  # [BV, D]
        b_state_tile = tl.load(state_ptr + p_state).to(tl.float32)  # [BV, D]

        # Apply gate decay
        b_gated_tile = b_g * b_state_tile  # [BV, D]

        # old_v[v] = sum_k(gated_state[v, k] * k[k])
        b_old_v_tile = tl.sum(b_gated_tile * b_k[None, :], axis=1)  # [BV]

        # Delta rule
        b_new_v_tile = b_beta * b_v_tile + (1.0 - b_beta) * b_old_v_tile  # [BV]
        b_delta_tile = b_new_v_tile - b_old_v_tile  # [BV]

        # Update state: new_state[v, k] = gated_state[v, k] + delta[v] * k[k]
        b_new_state_tile = b_gated_tile + b_delta_tile[:, None] * b_k[None, :]  # [BV, D]

        # Store new state tile
        tl.store(new_state_ptr + p_state, b_new_state_tile.to(new_state_ptr.dtype.element_ty))

        # Compute output[v] = scale * sum_k(q[k] * new_state[v, k])
        b_out_tile = scale * tl.sum(b_new_state_tile * b_q[None, :], axis=1)  # [BV]

        # Store output tile
        tl.store(out_ptr + qkv_base + offs_v, b_out_tile.to(out_ptr.dtype.element_ty))


@torch.no_grad()
def run(
    q: torch.Tensor,       # [B, 1, HQ, D] bf16
    k: torch.Tensor,       # [B, 1, HK, D] bf16
    v: torch.Tensor,       # [B, 1, HV, D] bf16
    state: Optional[torch.Tensor],  # [B, HV, D, D] f32, k-last
    A_log: torch.Tensor,   # [HV] f32
    a: torch.Tensor,       # [B, 1, HV] bf16
    dt_bias: torch.Tensor, # [HV] f32
    b: torch.Tensor,       # [B, 1, HV] bf16
    scale: Optional[float],
):
    B_size = q.shape[0]
    D = HEAD_SIZE
    HV = NUM_V_HEADS
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    # Compute gate and beta from raw parameters
    x = a.float() + dt_bias.float()           # [B, 1, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
    beta = torch.sigmoid(b.float())            # [B, 1, HV]

    # Squeeze out seq_len=1 dim and expand q/k heads to match v heads
    q_f = q.squeeze(1).float()  # [B, HQ, D]
    k_f = k.squeeze(1).float()  # [B, HK, D]
    v_f = v.squeeze(1).float()  # [B, HV, D]
    g_f = g.squeeze(1)          # [B, HV]
    beta_f = beta.squeeze(1)    # [B, HV]

    # Expand q and k heads: HQ=4 -> HV=8 (repeat_interleave by 2)
    q_exp = q_f.repeat_interleave(HEAD_EXPANSION, dim=1).contiguous()  # [B, HV, D]
    k_exp = k_f.repeat_interleave(HEAD_EXPANSION, dim=1).contiguous()  # [B, HV, D]

    # Handle state
    if state is not None:
        state_f32 = state.float().contiguous()  # [B, HV, D, D]
    else:
        state_f32 = torch.zeros(B_size, HV, D, D, dtype=torch.float32, device=device)

    # Allocate outputs
    output_flat = torch.empty(B_size, HV, D, dtype=torch.bfloat16, device=device)
    new_state = torch.empty_like(state_f32)

    # Launch kernel: one program per (batch, head)
    grid = (B_size * HV,)
    BV = 32  # tile size along V; D=128 / BV=32 = 4 iterations

    gdn_decode_kernel[grid](
        q_exp, k_exp, v_f.contiguous(),
        g_f.contiguous(), beta_f.contiguous(),
        state_f32,
        output_flat, new_state,
        scale,
        B=B_size,
        H=HV,
        D=D,
        BV=BV,
    )

    # Reshape output: [B, HV, D] -> [B, 1, HV, D]
    output = output_flat.unsqueeze(1)
    return output, new_state