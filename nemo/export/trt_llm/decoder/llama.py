# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The LLAMA/LLAMA2 decoder implementation."""

from typing import Optional

from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import AttentionMaskType, MoeConfig, PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.llama.model import LLaMADecoderLayer
from tensorrt_llm.models.modeling_utils import PretrainedConfig
from tensorrt_llm.quantization import QuantMode
from typing_extensions import override

from ..model_config import LINEAR_COLUMN, LINEAR_ROW, AttentionConfig, LayernormConfig, LinearConfig, MLPConfig
from .decoder import DecoderLayerBuilder, DecoderLayerConfigBuilder


class LLAMADecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    """The LLAMA implementation of the DecoderLayerConfigBuilder."""

    @override
    def hidden_act_fn(self, layer):
        return layer.mlp.act_fn

    @override
    def infer_num_attention_heads(self, layer):
        return layer.self_attn.num_heads

    @override
    def infer_num_kv_heads(self, layer):
        return layer.self_attn.num_key_value_heads

    @override
    def infer_max_position_embeddings(self, layer):
        return layer.self_attn.max_position_embeddings

    @override
    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.input_layernorm, dtype=self.dtype)

    @override
    def build_attention(self, layer) -> AttentionConfig:
        config = AttentionConfig()
        config.qkv = LinearConfig.from_qkv_nn_modules(
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.self_attn.o_proj, LINEAR_ROW, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        return config

    @override
    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.gate_proj, LINEAR_COLUMN, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.down_proj, LINEAR_ROW, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )
        config.gate = LinearConfig.from_nn_module(
            layer.mlp.up_proj, LINEAR_COLUMN, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        return config

    @override
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.post_attention_layernorm, dtype=self.dtype)


class LLAMADecoderLayerBuilder(DecoderLayerBuilder):
    """The LLAMA implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        rotary_scaling = None
        if layer.rotary_scaling is not None:
            rotary_scaling = {"type": "linear", "factor": float(layer.rotary_scaling)}

        config = PretrainedConfig(
            architecture=None,
            dtype=self.dtype,
            logits_dtype=self.dtype,
            vocab_size=layer.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.hidden_act.split("-")[-1] if layer.moe_num_experts else non_gated_version(self.hidden_act),
            intermediate_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            norm_epsilon=layer.norm_epsilon,
            position_embedding_type="rope_gpt_neox",
            world_size=self.tensor_parallel,
            tp_size=self.tensor_parallel,
            pp_size=1,
            quant_mode=QuantMode(0),
            quant_kwargs=None,
            # use_prompt_tuning=layer.use_prompt_tuning,
            max_lora_rank=layer.max_lora_rank,
            # use_parallel_embedding: bool = False,
            # embedding_sharding_dim: int = 0,
            # share_embedding_table: bool = False,
            # head_size: int = None,
        )

        config.set_if_not_exist('mlp_bias', False)
        config.set_if_not_exist('attn_bias', False)
        config.set_if_not_exist('rotary_base', layer.rotary_base)
        config.set_if_not_exist('rotary_scaling', rotary_scaling)
        config.set_if_not_exist('enable_pos_shift', False)
        config.set_if_not_exist('dense_context_fmha', False)
        config.set_if_not_exist('moe_num_experts', 0)

        if layer.moe_num_experts:
            moe_config = MoeConfig()
            if not layer.moe_num_experts is None:
                if layer.moe_top_k is None:
                    layer.moe_top_k = 1

                layer.moe_tp_mode = MoeConfig.ParallelismMode.TENSOR_PARALLEL if layer.moe_tp_mode is None else None
                layer.moe_renorm_mode = (
                    MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE if layer.moe_renorm_mode is None else None
                )
                moe_config = MoeConfig(
                    layer.moe_num_experts, layer.moe_top_k, layer.moe_tp_mode, layer.moe_renorm_mode
                ).validate()
                config.moe_num_experts = layer.moe_num_experts
                config.moe_top_k = layer.moe_top_k
                config.moe_tp_mode = layer.moe_tp_mode
                config.moe_normalization_mode = layer.moe_renorm_mode

        return LLaMADecoderLayer(config=config, layer_idx=self.layer_id,)
