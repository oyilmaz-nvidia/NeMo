# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utilities for exporting a model to our custom format."""

import numpy as np
import torch
from tensorrt_llm._utils import torch_to_numpy

# AMMO modification
# A global dicts to store exported weights.
# This is set to be a global variable to avoid extra code modification from tensorrt_llm.
weights_dict = {}


def cpu_map_location(storage, loc):
    return storage.cpu()


def gpu_map_location(storage, loc):
    if loc.startswith("cuda"):
        training_gpu_idx = int(loc.split(":")[1])
        inference_gpu_idx = training_gpu_idx % torch.cuda.device_count()
        return storage.cuda(inference_gpu_idx)
    elif loc.startswith("cpu"):
        return storage.cpu()
    else:
        raise ValueError(f"Not handled {loc}")


def save_val(val, dir, key, tp_num=None):
    suffix = "bin" if tp_num is None else f"{tp_num}.bin"
    # AMMO modification, save to in-memory dict instead of dir.
    # Transpose linear layer weights to the correct shape.
    if len(val.shape) >= 2:
        val = np.ascontiguousarray(np.transpose(val.reshape(val.shape[0], -1), [1, 0]))
    global weights_dict
    weights_dict[f"model.{key}.{suffix}"] = val


def save_split(split_vals, dir, key, i, split_factor):
    for j, val in enumerate(split_vals):
        save_val(val, dir, key, i * split_factor + j)


def generate_int8(weights, act_range, is_qkv=False, multi_query_mode=False):
    """This function has two purposes:
    - compute quantized weights, scaled either per-tensor or per-column
    - compute scaling factors.

    Depending on the GEMM API (CUTLASS/CUBLAS) the required scaling factors differ.
    CUTLASS uses two sets of scaling factors. One for the activation X, one for the weight W.
    CUBLAS only has one (we can't do per-row scaling). So we must provide pre-multiplied scaling factor.

    Here is the list of what we need (T means per-tensor, C per-column):
    - scale_x_orig_quant puts fp activation into the quantized range (i.e. [-128, 127], for int8).
    Used before the GEMM. (T)
    - scale_y_quant_orig puts quantized activation into the fp range. Used if the GEMM outputs int8. (T)
    - scale_w_quant_orig puts weights from quant range to fp range (used with CUTLASS) (T, C)
    - scale_y_accum_quant puts the GEMM result (XW) from accumulation range (int32)
    to quant range (int8) (used for CUBLAS) (T, C)

    Note that we don't do anything special about row-parallel GEMM.
    Theoretically, we could have per-GPU scaling factors too,
    but then the model would change depending on the number of GPUs used.

    For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection,
    we consider it
    as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
    """
    # compute weight scaling factors for fp->int8 and int8->fp
    if is_qkv and not multi_query_mode:
        scale_w_orig_quant_t = (
            127.0 / act_range["w"].reshape(3, -1).max(dim=-1, keepdims=True)[0].cpu().numpy()
        )
        scale_w_orig_quant_c = 127.0 / act_range["w"].reshape(3, -1).cpu().numpy()
    elif is_qkv and multi_query_mode:
        raise ValueError("Multi-query w/ int8 quant has not been supported yet")
    else:
        scale_w_orig_quant_t = 127.0 / act_range["w"].max().cpu().numpy()
        scale_w_orig_quant_c = 127.0 / act_range["w"].cpu().numpy()
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127.0 / act_range["x"].max().item())
    scale_y_orig_quant_t = np.array(127.0 / act_range["y"].max().item())
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.0)
    scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t * scale_w_orig_quant_t)
    scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t * scale_w_orig_quant_c)
    if is_qkv:
        scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t, scale_w_orig_quant_c.shape)
        scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t, scale_w_orig_quant_c.shape)

    def to_i8(x):
        return x.round().clip(-127, 127).astype(np.int8)

    return {
        "weight.int8": to_i8(weights * scale_w_orig_quant_t),
        "weight.int8.col": to_i8(weights * scale_w_orig_quant_c),
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


def write_int8(vals, dir, base_key, split_dim, tp_rank, split_factor, kv_cache_only=False):
    if not kv_cache_only:
        save_split(
            np.split(vals["weight.int8"], split_factor, axis=split_dim),
            dir,
            f"{base_key}.weight.int8",
            tp_rank,
            split_factor,
        )
        save_split(
            np.split(vals["weight.int8.col"], split_factor, axis=split_dim),
            dir,
            f"{base_key}.weight.int8.col",
            tp_rank,
            split_factor,
        )

    saved_keys_once = ["scale_y_quant_orig"]
    if not kv_cache_only:
        saved_keys_once += ["scale_x_orig_quant", "scale_w_quant_orig", "scale_y_accum_quant"]
    # per-column scaling factors are loaded per-gpu for ColumnParallel GEMMs (QKV, FC1)
    if not kv_cache_only:
        if split_dim == -1:
            save_split(
                np.split(vals["scale_w_quant_orig.col"], split_factor, axis=split_dim),
                dir,
                f"{base_key}.scale_w_quant_orig.col",
                tp_rank,
                split_factor,
            )
            save_split(
                np.split(vals["scale_y_accum_quant.col"], split_factor, axis=split_dim),
                dir,
                f"{base_key}.scale_y_accum_quant.col",
                tp_rank,
                split_factor,
            )
        else:
            saved_keys_once += ["scale_w_quant_orig.col", "scale_y_accum_quant.col"]

    if tp_rank == 0:
        for save_key in saved_keys_once:
            save_val(vals[save_key], dir, f"{base_key}.{save_key}")


# Note: in multi_query_mode, only query heads are split between multiple GPUs, while key/value head
# are not split as there is only one head per key/value.
@torch.no_grad()
def split_and_save_weight(
    tp_rank, saved_dir, split_factor, key, vals, storage_type, act_range, config
):
    use_attention_nemo_shape = config.get("use_attention_nemo_shape", False)
    split_gated_activation = config.get("split_gated_activation", False)
    num_attention_heads = config.get("num_attention_heads", 0)
    tp_size = config.get("tp_size", 1)
    int8_outputs = config.get("int8_outputs", None)
    multi_query_mode = config.get("multi_query_mode", False)
    local_dim = config.get("local_dim", None)
    num_kv_heads = config.get("num_kv_heads", num_attention_heads)

    save_int8 = int8_outputs == "all" or int8_outputs == "kv_cache_only"

    if not isinstance(vals, list):
        vals = [vals]

    if config.get("transpose_weights", False) and vals[0].ndim == 2:
        vals = [val.T for val in vals]
    if "layernorm.weight" in key and config.get("apply_layernorm_1p", False):
        vals = [val + 1.0 for val in vals]
    vals = [torch_to_numpy(val.cpu().to(storage_type)) for val in vals]

    if (
        "input_layernorm.weight" in key
        or "input_layernorm.bias" in key
        or "attention.dense.bias" in key
        or "post_attention_layernorm.weight" in key
        or "post_attention_layernorm.bias" in key
        or "post_self_attn_layernorm.weight" in key
        or "mlp.dense_4h_to_h.bias" in key
        or "final_layernorm.weight" in key
        or "final_layernorm.bias" in key
    ):
        # shared weights, only need to convert the weights of rank 0
        if "post_self_attn_layernorm.weight" in key:
            key = key.replace("post_self_attn_layernorm.weight", "post_attention_layernorm.weight")
        if tp_rank == 0:
            save_val(vals[0], saved_dir, key)

    elif (
        "attention.dense.weight" in key
        or "mlp.dense_4h_to_h.weight" in key
        or "attention.linear_proj.weight" in key
        or "mlp.linear_fc2.weight" in key
    ):
        cat_dim = 0
        val = np.concatenate(vals, axis=cat_dim)
        split_vals = np.split(val, split_factor, axis=cat_dim)
        if "attention.linear_proj.weight" in key:
            key = key.replace("attention.linear_proj.weight", "attention.dense.weight")
        elif "mlp.linear_fc2.weight" in key:
            key = key.replace("mlp.linear_fc2.weight", "mlp.dense_4h_to_h.weight")
        save_split(split_vals, saved_dir, key, tp_rank, split_factor)
        if act_range is not None and int8_outputs == "all":
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, multi_query_mode=multi_query_mode)
            write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank, split_factor)

    elif (
        "mlp.dense_h_to_4h.weight" in key
        or "mlp.dense_h_to_4h.bias" in key
        or "mlp.linear_fc1.weight" in key
    ):
        if split_gated_activation:
            splits = [np.split(val, 2, axis=-1) for val in vals]
            vals, gates = list(zip(*splits))
        cat_dim = -1
        val = np.concatenate(vals, axis=cat_dim)
        split_vals = np.split(val, split_factor, axis=cat_dim)
        if "mlp.linear_fc1.weight" in key:
            key = key.replace("mlp.linear_fc1.weight", "mlp.dense_h_to_4h.weight")
        save_split(split_vals, saved_dir, key, tp_rank, split_factor)
        if act_range is not None and int8_outputs == "all":
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, multi_query_mode=multi_query_mode)
            write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank, split_factor)

        if split_gated_activation:
            assert not save_int8
            prefix, dot, suffix = key.rpartition(".")
            key = prefix + ".gate" + dot + suffix

            gate = np.concatenate(gates, axis=cat_dim)
            split_vals = np.split(gate, split_factor, axis=cat_dim)
            save_split(split_vals, saved_dir, key, tp_rank, split_factor)

    # Ammo modification
    elif "mlp.dense_h_to_4h_2.weight" in key or "mlp.dense_h_to_4h_2.bias" in key:
        cat_dim = -1
        val = np.concatenate(vals, axis=cat_dim)
        split_vals = np.split(val, split_factor, axis=cat_dim)
        save_split(split_vals, saved_dir, key, tp_rank, split_factor)
        if act_range is not None and int8_outputs == "all":
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, multi_query_mode=multi_query_mode)
            write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank, split_factor)

    elif "attention.query_key_value.bias" in key:
        assert (
            num_attention_heads == num_kv_heads or multi_query_mode
        ), "QKV bias is not supported for group query attention"
        if local_dim is None:
            local_dim = vals[0].shape[-1] // 3

        if multi_query_mode:
            val = vals[0]
            # out_feature = local_dim + 2 * head_size; assumes local_dim equals to hidden_dim
            b_q, b_kv = np.split(val, [local_dim], axis=-1)
            b_q_split = np.split(b_q, split_factor, axis=-1)
            split_vals = [np.concatenate((i, b_kv), axis=-1) for i in b_q_split]
        else:
            if use_attention_nemo_shape:
                head_num = num_attention_heads // tp_size
                size_per_head = local_dim // num_attention_heads
                nemo_shape = (head_num, 3, size_per_head)
                vals = [val.reshape(nemo_shape) for val in vals]
                vals = [val.transpose(1, 0, 2) for val in vals]

            vals = [val.reshape(3, local_dim) for val in vals]
            val = np.concatenate(vals, axis=-1)
            split_vals = np.split(val, split_factor, axis=-1)
        save_split(split_vals, saved_dir, key, tp_rank, split_factor)

    elif "attention.query_key_value.weight" in key or "attention.linear_qkv.weight" in key:
        assert use_attention_nemo_shape, "Only support NEMO shape for QKV weights"
        hidden_dim = vals[0].shape[0]
        size_per_head = hidden_dim // num_attention_heads
        q_num = num_attention_heads // num_kv_heads
        merge_factor = len(vals)

        # When the merge factor exceeds 1, the 'vals' list will have multiple entries.
        # Depending on the format, 'vals' can look like either [QQQQ..KV, QQQQ..KV, ...](for GQA) or [QKV, QKV, ...](for MHA).
        # We transform 'vals' into a unified structure: [QQQQQ...KK..VV..].
        # If the split factor surpasses 1, this array undergoes a split along its last dimension, which is 'size_per_head'.

        # Reshape each elements of the vals array to shape:
        # (hidden_dim, q_num + 2, num_kv_heads // tp_size, size_per_head)
        vals = [
            val.reshape(hidden_dim, num_kv_heads // tp_size, q_num + 2, size_per_head).transpose(
                0, 2, 1, 3
            )
            for val in vals
        ]

        # Combine all the Qs, Ks and Vs together for each val
        q_splits, k_splits, v_splits = zip(
            *[np.split(val, [q_num, q_num + 1], axis=1) for val in vals]
        )

        # Concatenate Q, K, and V separately
        q_splits_concat = np.concatenate(q_splits, axis=1)
        k_splits_concat = np.concatenate(k_splits, axis=1)
        v_splits_concat = np.concatenate(v_splits, axis=1)

        # Concatenate Q, K, and V together and reshape
        qkv_split_concat = np.concatenate(
            [q_splits_concat, k_splits_concat, v_splits_concat], axis=1
        )
        qkv_split_concat = qkv_split_concat.reshape(
            hidden_dim, q_num + 2, (num_kv_heads // tp_size) * size_per_head * merge_factor
        )

        # Final split
        split_vals = np.split(qkv_split_concat, split_factor, axis=-1)
        if "attention.linear_qkv.weight" in key:
            key = key.replace("attention.linear_qkv.weight", "attention.query_key_value.weight")
        save_split(split_vals, saved_dir, key, tp_rank, split_factor)
        if save_int8:
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, is_qkv=True, multi_query_mode=multi_query_mode)
            write_int8(
                vals_i8,
                saved_dir,
                base_key,
                cat_dim,
                tp_rank,
                split_factor,
                kv_cache_only=int8_outputs == "kv_cache_only",
            )
    elif (
        "attention.query.weight" in key
        or "attention.query.bias" in key
        or "attention.key_value.weight" in key
        or "attention.key_value.bias" in key
    ):
        pass
    else:
        print(f"[WARNING] {key} not handled by converter")

    # Ammo modification
    global weights_dict
    return weights_dict
