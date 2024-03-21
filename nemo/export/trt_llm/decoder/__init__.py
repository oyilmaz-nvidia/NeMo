# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LLM Decoder implementation for tensorrt_llm conversion."""
from typing import Dict, Type

import tensorrt as trt

from ..model_config import (
    DECODER_FALCON,
    DECODER_GEMMA,
    DECODER_GPT2,
    DECODER_GPTJ,
    DECODER_GPTNEXT,
    DECODER_LLAMA,
    QUANTIZATION_NONE,
)
from .decoder import DecoderLayerBuilder, DecoderLayerConfigBuilder
from .falcon import FALCONDecoderLayerBuilder, FALCONDecoderLayerConfigBuilder
from .gemma import GemmaDecoderLayerBuilder, GemmaDecoderLayerConfigBuilder
from .gpt import GPTDecoderLayerBuilder, GPTDecoderLayerConfigBuilder
from .gptj import GPTJDecoderLayerBuilder, GPTJDecoderLayerConfigBuilder
from .llama import LLAMADecoderLayerBuilder, LLAMADecoderLayerConfigBuilder

DECODER_CONFIG_REGISTRY: Dict[str, Type[DecoderLayerConfigBuilder]] = {
    DECODER_GPT2: GPTDecoderLayerConfigBuilder,
    DECODER_GPTJ: GPTJDecoderLayerConfigBuilder,
    DECODER_LLAMA: LLAMADecoderLayerConfigBuilder,
    DECODER_FALCON: FALCONDecoderLayerConfigBuilder,
    DECODER_GEMMA: GemmaDecoderLayerConfigBuilder,
}


def build_decoder_layer_config(layer, decoder: str, dtype=trt.float16, rank=0, tensor_parallel=1):
    """Builds the decoder layer config with the input torch module."""
    assert decoder in DECODER_CONFIG_REGISTRY, f"{decoder} not supported"
    return DECODER_CONFIG_REGISTRY[decoder](decoder, dtype, rank, tensor_parallel).build_layer(layer)


DECODER_REGISTRY: Dict[str, Type[DecoderLayerBuilder]] = {
    DECODER_GPT2: GPTDecoderLayerBuilder,
    DECODER_GPTJ: GPTJDecoderLayerBuilder,
    DECODER_LLAMA: LLAMADecoderLayerBuilder,
    DECODER_GPTNEXT: GPTDecoderLayerBuilder,
    DECODER_FALCON: FALCONDecoderLayerBuilder,
    DECODER_GEMMA: GemmaDecoderLayerBuilder,
}


def build_decoder_layer(
    layer,
    layer_id: int,
    num_layers: int,
    dtype=trt.float16,
    quantization=QUANTIZATION_NONE,
    rank=0,
    tensor_parallel=1,
    tp_group=None,
):
    """Builds the tensorrt llm decoder layer module with the layer config as the input."""
    assert layer.decoder_type in DECODER_REGISTRY, f"{layer.decoder_type} not supported"
    builder = DECODER_REGISTRY[layer.decoder_type]
    decoder_builder = builder(layer, layer_id, num_layers, dtype, quantization, rank, tensor_parallel, tp_group)
    return decoder_builder.decoder
