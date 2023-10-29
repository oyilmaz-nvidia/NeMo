# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The FALCON decoder implementation."""

from typing import Optional

from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.models.falcon.model import FalconDecoderLayer 
from typing_extensions import override

from ..model_config import (
    LINEAR_COLUMN,
    LINEAR_ROW,
    AttentionConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
)
from .decoder import DecoderLayerBuilder, DecoderLayerConfigBuilder


class FALCONDecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    """The FALCON implementation of the DecoderLayerConfigBuilder."""

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
    def build_mlp_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.mlp_layernorm, dtype=self.dtype)

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
            layer.self_attn.o_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    @override
    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.gate_proj,
            LINEAR_COLUMN,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.down_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )
        config.gate = LinearConfig.from_nn_module(
            layer.mlp.up_proj,
            LINEAR_COLUMN,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    @override
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.post_attention_layernorm, dtype=self.dtype)


class FALCONDecoderLayerBuilder(DecoderLayerBuilder):
    """The FALCON implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        #Falcon 7B: parallel_attention=True, new_decoder_architecture=False
        #Falcon 40B/180B: parallel_attention=True, new_decoder_architecture=True
        flayer = FalconDecoderLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_kv_heads=self.num_kv_heads,
            dtype=self.dtype,
            hidden_act=non_gated_version(self.hidden_act),
            mlp_hidden_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            bias=False, # False from HF repo config
            use_alibi=False,
            new_decoder_architecture=False if self.num_layers==32 else True, # No other way to pass in model variant config, determine model variant by num_layers (7B: 32 layers)
            parallel_attention=True,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            layer_id=self.layer_id,
            layernorm_epsilon=1e-5,
        )

        return flayer
