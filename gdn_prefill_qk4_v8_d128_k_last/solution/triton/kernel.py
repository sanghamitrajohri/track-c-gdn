# Forward-only standalone chunked Gated DeltaNet kernel specialized to the contest ABI.
# Ablation:
# - based on `chunk_fused_inverse_apply_s3.py`
# - keeps persistent state in external contest `[V, K]` layout throughout
# - removes both host-side state transpose copies
# - changes the hot kernels to operate on the transposed-state path
# - refresh: uses vectorized CPU chunk-index metadata, Triton fused `g_log`/`beta` preprocessing,
#   and a wrapper fastpath that skips redundant validation / `.contiguous()` work

import functools
import inspect
import math
import os
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": os.getenv("FLA_CACHE_RESULTS", "1") == "1"} if SUPPORTS_AUTOTUNE_CACHE else {}

IS_NVIDIA = torch.cuda.is_available()
USE_CUDA_GRAPH = IS_NVIDIA and os.getenv("FLA_USE_CUDA_GRAPH", "0") == "1"
IS_TMA_SUPPORTED = (
    IS_NVIDIA
    and torch.cuda.get_device_capability(0)[0] >= 9
    and os.getenv("FLA_USE_TMA", "0") == "1"
    and (hasattr(triton.language, "_experimental_make_tensor_descriptor") or hasattr(triton.language, "make_tensor_descriptor"))
)
DOT_PRECISION_AUTOTUNE_LIST = ["ieee"]

NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_SIZE = 128
HEAD_EXPANSION = NUM_V_HEADS // NUM_Q_HEADS

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    @triton.jit
    def exp(x):
        return tldevice.fast_expf(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tldevice.exp2(x.to(tl.float32))
else:
    @triton.jit
    def exp(x):
        return tl.exp(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tl.math.exp2(x.to(tl.float32))

if hasattr(triton.language, '_experimental_make_tensor_descriptor'):
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, 'make_tensor_descriptor'):
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        return None


def tensor_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    last_args: tuple | None = None
    last_kwargs: dict | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result
        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args, strict=False)) and                         all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result
        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def input_guard(
    fn: Callable[..., Any] | None = None,
    *,
    no_guard_contiguous: bool | list[str] = False,
):
    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(inner)
        param_names = list(sig.parameters.keys())

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            skip_params = set(no_guard_contiguous) if isinstance(no_guard_contiguous, list) else set()
            processed_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else f"__arg_{i}"
                if isinstance(arg, torch.Tensor) and not (no_guard_contiguous is True or param_name in skip_params):
                    processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)
            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and not (no_guard_contiguous is True or k in skip_params):
                    processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v
            return inner(*processed_args, **processed_kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


class _Backend:
    DEFAULT = 102400
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        max_shared_mem = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)["max_shared_mem"]
    except Exception:
        props = torch.cuda.get_device_properties(tensor_idx)
        max_shared_mem = getattr(props, "shared_memory_per_block_optin", getattr(props, "shared_memory_per_block", 0))
    target = {
        "ada": _Backend.ADA,
        "ampere": _Backend.AMPERE,
        "hopper": _Backend.HOPPER,
        "none": _Backend.DEFAULT,
    }.get(arch.lower(), _Backend.DEFAULT)
    return max_shared_mem >= target


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)

@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    base = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
    lens = torch.diff(base)
    n_chunks = (lens + chunk_size - 1) // chunk_size
    total = int(n_chunks.sum().item())
    if total == 0:
        return torch.empty((0, 2), device=cu_seqlens.device, dtype=torch.int32)
    seq_ids = torch.repeat_interleave(
        torch.arange(n_chunks.numel(), dtype=base.dtype, device=base.device),
        n_chunks,
        output_size=total,
    )
    starts = torch.cat(
        [
            torch.zeros(1, dtype=base.dtype, device=base.device),
            n_chunks.cumsum(0)[:-1],
        ]
    )
    chunk_ids = torch.arange(total, dtype=base.dtype, device=base.device) - torch.repeat_interleave(
        starts, n_chunks, output_size=total
    )
    out = torch.stack((seq_ids, chunk_ids), dim=1)
    return out.to(device=cu_seqlens.device, dtype=torch.int32)

@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_kh = i_h // 2
    H_K = H // 2
    i_kh = i_h // 2
    H_K = H // 2
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))

def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_kh = i_h // 2
    H_K = H // 2
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_A_33 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_A_44 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_A_31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_A_32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_A_41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_A_42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_A_43 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
        b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
        b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
        b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
        b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne"))

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_kh = i_h // 2
    H_K = H // 2
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))
    p_b1 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,))
    p_b4 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
    b_b4 = tl.load(p_b4, boundary_check=(0,)).to(tl.float32)
    p_b1 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,))
    p_b4 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
    b_b4 = tl.load(p_b4, boundary_check=(0,)).to(tl.float32)
    p_b1 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,))
    p_b4 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
    b_b4 = tl.load(p_b4, boundary_check=(0,)).to(tl.float32)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H_K + i_kh) * K, (T, K), (H_K*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp(b_g_diff)
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H_K, K = k.shape
    H = beta.shape[-1]
    assert H == H_K * HEAD_EXPANSION
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return A

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3)
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_kh = i_h // 2
    H_K = H // 2
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))
    p_b1 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 0,), (16,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 16,), (16,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 32,), (16,), (0,))
    p_b4 = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT + 48,), (16,), (0,))
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
    b_b4 = tl.load(p_b4, boundary_check=(0,)).to(tl.float32)

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A += (bos * H + i_h) * BT
    p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, allow_tf32=False), b_Ai_11, allow_tf32=False)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, allow_tf32=False), b_Ai_22, allow_tf32=False)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, allow_tf32=False), b_Ai_33, allow_tf32=False)
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, allow_tf32=False) + tl.dot(b_A_32, b_Ai_21, allow_tf32=False),
        allow_tf32=False,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, allow_tf32=False) + tl.dot(b_A_43, b_Ai_32, allow_tf32=False),
        allow_tf32=False,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, allow_tf32=False) +
        tl.dot(b_A_42, b_Ai_21, allow_tf32=False) +
        tl.dot(b_A_43, b_Ai_31, allow_tf32=False),
        allow_tf32=False,
    )

    for i_v in range(tl.cdiv(V, BV)):
        p_v1 = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + 0, i_v * BV), (16, BV), (1, 0))
        p_v2 = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0))
        p_v3 = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0))
        p_v4 = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0))
        p_u1 = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + 0, i_v * BV), (16, BV), (1, 0))
        p_u2 = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0))
        p_u3 = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0))
        p_u4 = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0))
        b_v1 = (tl.load(p_v1, boundary_check=(0, 1)) * b_b1[:, None]).to(tl.float32)
        b_v2 = (tl.load(p_v2, boundary_check=(0, 1)) * b_b2[:, None]).to(tl.float32)
        b_v3 = (tl.load(p_v3, boundary_check=(0, 1)) * b_b3[:, None]).to(tl.float32)
        b_v4 = (tl.load(p_v4, boundary_check=(0, 1)) * b_b4[:, None]).to(tl.float32)
        b_u1 = tl.dot(b_Ai_11, b_v1, allow_tf32=False)
        b_u2 = tl.dot(b_Ai_21, b_v1, allow_tf32=False) + tl.dot(b_Ai_22, b_v2, allow_tf32=False)
        b_u3 = tl.dot(b_Ai_31, b_v1, allow_tf32=False) + tl.dot(b_Ai_32, b_v2, allow_tf32=False) + tl.dot(b_Ai_33, b_v3, allow_tf32=False)
        b_u4 = tl.dot(b_Ai_41, b_v1, allow_tf32=False) + tl.dot(b_Ai_42, b_v2, allow_tf32=False) + tl.dot(b_Ai_43, b_v3, allow_tf32=False) + tl.dot(b_Ai_44, b_v4, allow_tf32=False)
        tl.store(p_u1, b_u1.to(p_u1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u2, b_u2.to(p_u2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u3, b_u3.to(p_u3.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_u4, b_u4.to(p_u4.dtype.element_ty), boundary_check=(0, 1))

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = exp(tl.load(p_g, boundary_check=(0,)))
        p_g1 = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT + 0,), (16,), (0,))
        p_g2 = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT + 16,), (16,), (0,))
        p_g3 = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT + 32,), (16,), (0,))
        p_g4 = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT + 48,), (16,), (0,))
        b_g1 = exp(tl.load(p_g1, boundary_check=(0,))).to(tl.float32)
        b_g2 = exp(tl.load(p_g2, boundary_check=(0,))).to(tl.float32)
        b_g3 = exp(tl.load(p_g3, boundary_check=(0,))).to(tl.float32)
        b_g4 = exp(tl.load(p_g4, boundary_check=(0,))).to(tl.float32)
        p_g1 = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT + 0,), (16,), (0,))
        p_g2 = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT + 16,), (16,), (0,))
        p_g3 = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT + 32,), (16,), (0,))
        p_g4 = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT + 48,), (16,), (0,))
        b_g1 = exp(tl.load(p_g1, boundary_check=(0,))).to(tl.float32)
        b_g2 = exp(tl.load(p_g2, boundary_check=(0,))).to(tl.float32)
        b_g3 = exp(tl.load(p_g3, boundary_check=(0,))).to(tl.float32)
        b_g4 = exp(tl.load(p_g4, boundary_check=(0,))).to(tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k1 = tl.make_block_ptr(k + (bos*H_K + i_kh) * K, (T, K), (H_K*K, 1), (i_t * BT + 0, i_k * BK), (16, BK), (1, 0))
        p_k2 = tl.make_block_ptr(k + (bos*H_K + i_kh) * K, (T, K), (H_K*K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0))
        p_k3 = tl.make_block_ptr(k + (bos*H_K + i_kh) * K, (T, K), (H_K*K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0))
        p_k4 = tl.make_block_ptr(k + (bos*H_K + i_kh) * K, (T, K), (H_K*K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0))
        p_w1 = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 0, i_k * BK), (16, BK), (1, 0))
        p_w2 = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0))
        p_w3 = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0))
        p_w4 = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0))
        b_k1 = (tl.load(p_k1, boundary_check=(0, 1)) * b_b1[:, None]).to(tl.float32)
        b_k2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_b2[:, None]).to(tl.float32)
        b_k3 = (tl.load(p_k3, boundary_check=(0, 1)) * b_b3[:, None]).to(tl.float32)
        b_k4 = (tl.load(p_k4, boundary_check=(0, 1)) * b_b4[:, None]).to(tl.float32)
        if USE_G:
            b_k1 *= b_g1[:, None]
            b_k2 *= b_g2[:, None]
            b_k3 *= b_g3[:, None]
            b_k4 *= b_g4[:, None]
        b_w1 = tl.dot(b_Ai_11, b_k1, allow_tf32=False)
        b_w2 = tl.dot(b_Ai_21, b_k1, allow_tf32=False) + tl.dot(b_Ai_22, b_k2, allow_tf32=False)
        b_w3 = tl.dot(b_Ai_31, b_k1, allow_tf32=False) + tl.dot(b_Ai_32, b_k2, allow_tf32=False) + tl.dot(b_Ai_33, b_k3, allow_tf32=False)
        b_w4 = tl.dot(b_Ai_41, b_k1, allow_tf32=False) + tl.dot(b_Ai_42, b_k2, allow_tf32=False) + tl.dot(b_Ai_43, b_k3, allow_tf32=False) + tl.dot(b_Ai_44, b_k4, allow_tf32=False)
        tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w2, b_w2.to(p_w2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w3, b_w3.to(p_w3.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w4, b_w4.to(p_w4.dtype.element_ty), boundary_check=(0, 1))

def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H_K, K, V = *k.shape, v.shape[-1]
    H = beta.shape[-1]
    assert H == H_K * HEAD_EXPANSION
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty(B, T, H, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B*H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2] if check_shared_mem('ampere') else [2, 1])
        for BV in ([32, 64] if check_shared_mem('ada') else [32])
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2', 'TRANSPOSE_STATE'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    i_kh = i_h // 2
    H_K = H // 2
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K*V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H_K + i_kh).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_h2 *= exp(b_gk_last2)[None, :]
                else:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_h3 *= exp(b_gk_last3)[None, :]
                else:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_h4 *= exp(b_gk_last4)[None, :]
                else:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H_K*K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, H_K*K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, H_K*K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, H_K*K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H_K, K = k.shape
    H = u.shape[2]
    V = u.shape[-1]
    assert H == H_K * HEAD_EXPANSION
    BT = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)
    final_state = k.new_zeros(N, H, V, K, dtype=torch.float32)

    v_new = torch.empty_like(u)
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=None,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=False,
        TRANSPOSE_STATE=True,
    )
    return h, v_new, final_state

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'K', 'V', 'BT', 'TRANSPOSE_STATE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_qh = i_h // 2
    H_Q = H // 2

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H_Q + i_qh) * K
    k += (bos * H_Q + i_qh) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H_Q*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H_Q*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        if TRANSPOSE_STATE:
            b_o += tl.dot(b_q, tl.trans(b_h))
        else:
            b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1))
    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H_Q, K = q.shape
    H = v.shape[2]
    V = v.shape[-1]
    assert H == H_Q * HEAD_EXPANSION
    BT = 64
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=None,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=True,
    )
    return o

@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] == 64
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    merge_16x16_to_64x64_inverse_kernel[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=IS_TMA_SUPPORTED,
    )
    return Ai

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None = None,
):
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64) if cu_seqlens is not None else None
    g = chunk_local_cumsum_scalar(
        g=g,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float,
    )
    a_inv = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=a_inv,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return o, final_state

def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert A_log.is_cuda and a.is_cuda and dt_bias.is_cuda and b.is_cuda and cu_seqlens.is_cuda
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert A_log.dtype == torch.float32
    assert dt_bias.dtype == torch.float32
    assert a.dtype == torch.bfloat16
    assert b.dtype == torch.bfloat16
    if state is not None:
        assert state.is_cuda and state.dtype == torch.float32
    assert q.shape[1:] == (NUM_Q_HEADS, HEAD_SIZE)
    assert k.shape[1:] == (NUM_K_HEADS, HEAD_SIZE)
    assert v.shape[1:] == (NUM_V_HEADS, HEAD_SIZE)
    assert A_log.shape == (NUM_V_HEADS,)
    assert dt_bias.shape == (NUM_V_HEADS,)
    assert a.shape == (q.shape[0], NUM_V_HEADS)
    assert b.shape == (q.shape[0], NUM_V_HEADS)
    assert cu_seqlens.dtype == torch.int64
    assert int(cu_seqlens[-1].item()) == q.shape[0]
    if state is not None:
        assert state.shape == (cu_seqlens.numel() - 1, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE)


@triton.autotune(
    configs=[triton.Config({}, num_warps=4)],
    key=["T", "H"],
    **autotune_cache_kwargs,
)
@triton.jit
def preprocess_g_beta_kernel(
    a,
    b,
    A_log,
    dt_bias,
    g_out,
    beta_out,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_a = tl.make_block_ptr(a, (T, H), (H, 1), (i_t * BT, 0), (BT, H), (1, 0))
    p_b = tl.make_block_ptr(b, (T, H), (H, 1), (i_t * BT, 0), (BT, H), (1, 0))
    p_g = tl.make_block_ptr(g_out, (T, H), (H, 1), (i_t * BT, 0), (BT, H), (1, 0))
    p_beta = tl.make_block_ptr(beta_out, (T, H), (H, 1), (i_t * BT, 0), (BT, H), (1, 0))

    b_a = tl.load(p_a, boundary_check=(0, 1)).to(tl.float32)
    b_b = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
    h_ids = tl.arange(0, H)
    b_A_log = tl.load(A_log + h_ids, mask=h_ids < H, other=0.0).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + h_ids, mask=h_ids < H, other=0.0).to(tl.float32)

    x = b_a + b_dt_bias[None, :]
    softplus_x = tl.where(
        x > 0,
        x + tl.log(1.0 + tl.exp(-x)),
        tl.log(1.0 + tl.exp(x)),
    )
    b_g = -tl.exp(b_A_log)[None, :] * softplus_x
    b_beta = tl.sigmoid(b_b)

    tl.store(p_g, b_g.to(p_g.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_beta, b_beta.to(p_beta.dtype.element_ty), boundary_check=(0, 1))


def preprocess_g_beta(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, H = a.shape
    g_out = torch.empty((T, H), device=a.device, dtype=torch.float32)
    beta_out = torch.empty((T, H), device=a.device, dtype=torch.float32)
    preprocess_g_beta_kernel[(triton.cdiv(T, 128),)](
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        g_out=g_out,
        beta_out=beta_out,
        T=T,
        H=H,
        BT=128,
    )
    return g_out, beta_out


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
    cu_seqlens: torch.Tensor,
    scale: Optional[float],
):
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(HEAD_SIZE)
    g_log, beta = preprocess_g_beta(a, b, A_log, dt_bias)
    output, final_state = chunk_gated_delta_rule_fwd(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g_log.unsqueeze(0),
        beta=beta.unsqueeze(0),
        scale=float(scale),
        initial_state=state,
        cu_seqlens=cu_seqlens,
    )
    return output.squeeze(0), final_state
