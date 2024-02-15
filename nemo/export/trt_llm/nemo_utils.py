# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The APIs to convert a nemo model checkpoint to tensorrt_llm."""

import argparse
import ast
import configparser
import copy
import datetime
import logging
import os
import sys
import tempfile
from pathlib import Path
import typing
from typing import Dict, List, Tuple

import csv
import torch
import numpy as np
import tensorrt_llm
from tensorrt_llm import str_dtype_to_trt
from transformers import GPT2Config, LlamaConfig, PretrainedConfig, PreTrainedTokenizer, AutoTokenizer

from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    LINEAR_COLUMN,
    DecoderLayerConfig,
    EmbeddingConfig,
    LayernormConfig,
    LinearConfig,
    ModelConfig,
)
from .nemo.nemo import UnpackedNemoCheckpointDir, unpack_nemo_ckpt
from .nemo.nemo_ckpt_convert import build_tokenizer, convert_checkpoint, convert_dist_checkpoint, convert_nemo_model
from .tensor_utils import get_tensor_from_dict, split, get_tensor_parallel_group

LOGGER = logging.getLogger("NeMo")


def _nemo_decode(
    in_file: str,
    out_dir: str,
    tensor_parallelism: int = 1,
    processes: int = 1,
    storage_type: str = "bfloat16",
    load_checkpoints_on_gpu: bool = False,
    decoder_type: str = "gptnext",
) -> Tuple[Dict[str, np.ndarray], PretrainedConfig, PreTrainedTokenizer]:
    """Decodes the NEMO file and returns the weights dict, llm config and tokenizer."""
    args = argparse.Namespace()
    args.in_file = in_file
    args.out_dir = out_dir
    args.tensor_parallelism = tensor_parallelism
    args.processes = processes
    args.storage_type = storage_type
    args.load_checkpoints_on_gpu = load_checkpoints_on_gpu
    args.verbose = False
    args.decoder_type = decoder_type

    input_path = Path(args.in_file)
    if not input_path.exists():
        LOGGER.error("%s does not exists", input_path)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # unpack if needed
        if input_path.is_dir():
            nemo_dir = input_path
        else:
            start_time = datetime.datetime.now()
            checkpoint_dir_path = temp_dir / "unpacked"
            nemo_dir = unpack_nemo_ckpt(args.in_file, checkpoint_dir_path)
            LOGGER.info(
                "Spent %s (h:m:s) to unpack NeMo archive", datetime.datetime.now() - start_time
            )

        unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(
            nemo_dir, load_checkpoints_to_cpu=not args.load_checkpoints_on_gpu
        )

        start_time = datetime.datetime.now()
        dist_ckpt_folder = nemo_dir / "model_weights"

        if dist_ckpt_folder.exists():
            weights_dict, llm_config, tokenizer = convert_dist_checkpoint(unpacked_checkpoint_dir, args)
        else:
            weights_dict, llm_config, tokenizer = convert_checkpoint(unpacked_checkpoint_dir, args)
        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)

        return weights_dict, llm_config, tokenizer


def get_model_config(weights_dir: Path) -> GPT2Config:
    """Reads the GPT2Config from the decoded NEMO weights dir."""
    config = configparser.ConfigParser()
    config_path = weights_dir / "config.ini"
    assert os.path.isfile(config_path), f"{config_path} not present"
    config.read(config_path)
    config_dict = dict(config.items("gpt"))
    # Parse the config to dict.
    for k, v in config_dict.items():
        try:
            config_dict[k] = ast.literal_eval(v)
        except Exception:
            pass
    return GPT2Config(**config_dict)


def get_tokenzier(tokenizer_dir_or_path: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    if os.path.isdir(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer")):
        return AutoTokenizer.from_pretrained(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer"))

    model_path = (
        tokenizer_dir_or_path / "tokenizer.model"
        if tokenizer_dir_or_path.is_dir()
        else tokenizer_dir_or_path
    )
    tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
    return build_tokenizer(tokenizer_config)

def nemo_to_model_config(
    in_file: str, 
    decoder_type: str, 
    nemo_export_dir: typing.Union[str, Path], 
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> Tuple[List[ModelConfig], PreTrainedTokenizer]:
    """Converts the NEMO file and construct the `ModelConfig` before tensorrt_llm deployment."""
    dtype_str = dtype

    weights_dict, llm_model_config, tokenizer = _nemo_decode(
        in_file=in_file,
        out_dir=nemo_export_dir,
        tensor_parallelism=tensor_parallel_size,
        processes=1,
        storage_type=dtype_str,
        load_checkpoints_on_gpu=False,
        decoder_type=decoder_type,
    )

    world_size = tensor_parallel_size*pipeline_parallel_size
    model_config_template = ModelConfig()
    model_config_template.dtype = dtype_str

    str_dtype_to_trt(dtype_str)

    model_configs = []
    for i in range(world_size):
        
        model_configs.append(copy.deepcopy(model_config_template))

        model_configs[i].vocab_embedding = EmbeddingConfig(
            weight=get_tensor_from_dict(weights_dict, "wte")
        )

        model_configs[i].final_layernorm = LayernormConfig(
            weight=get_tensor_from_dict(weights_dict, "final_layernorm.weight"),
            bias=get_tensor_from_dict(weights_dict, "final_layernorm.bias"),
        )
        model_configs[i].final_layernorm.layernorm_type = (
            LAYERNORM_RMS if isinstance(llm_model_config, LlamaConfig) else LAYERNORM_DEFAULT
        )
        model_configs[i].mapping = tensorrt_llm.Mapping(
            world_size=world_size, 
            rank=i, 
            tp_size=tensor_parallel_size, 
            pp_size=pipeline_parallel_size)

    for i in range(llm_model_config.n_layer):
        for j in range(world_size):
            model_configs[j].layers.append(
                DecoderLayerConfig.from_nemo(
                    weights_dict=weights_dict,
                    llm_config=llm_model_config,
                    decoder_type=decoder_type,
                    layer_id=i,
                    rank=model_configs[j].mapping.tp_rank,
                    is_mcore=llm_model_config.is_mcore,
                )
            )

    lm_head_weight = get_tensor_from_dict(weights_dict, "lm_head.weight")

    if model_configs[0].vocab_size_padded != model_configs[0].vocab_size:
        pad_width = model_configs[0].vocab_size_padded - model_configs[0].vocab_size
        lm_head_weight = np.pad(
            lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0
        )

    for i in range(world_size):
        model_configs[i].lm_head = LinearConfig(linear_type=LINEAR_COLUMN)
        model_configs[i].lm_head.weight = np.ascontiguousarray(
            split(lm_head_weight, model_configs[i].mapping.tp_size, model_configs[i].mapping.tp_rank)
        )

    return model_configs, tokenizer

def to_word_list_format(word_dict: List[List[str]],
                        tokenizer=None):
    '''
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    '''
    assert tokenizer != None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(word)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))

def nemo_model_to_model_config(
    nemo_model: str, 
    decoder_type: str, 
    nemo_model_config: str, 
    dtype_str: str = "float32",
) -> Tuple[List[ModelConfig], PreTrainedTokenizer]:
    """Converts the NEMO model object and construct the `ModelConfig` before tensorrt_llm deployment."""
    from megatron.core import parallel_state
    assert nemo_model_config is not None, "gpt_model_config must be provided when in is a nemo model"

    weights_dict, llm_model_config = convert_nemo_model(
        nemo_model, nemo_model_config, dtype_str, decoder_type)
    is_mcore = nemo_model_config.get("mcore_gpt", False)
    llm_model_config.is_mcore = is_mcore

    model_config = ModelConfig()
    model_config.use_prompt_tuning = False
    model_config.dtype = dtype_str
    model_config.use_parallel_embedding = True
    str_dtype_to_trt(dtype_str)

    model_config.vocab_embedding = EmbeddingConfig(
        weight=get_tensor_from_dict(weights_dict, "wte"), is_local=True
    )

    model_config.final_layernorm = LayernormConfig(
        weight=get_tensor_from_dict(weights_dict, "final_layernorm.weight"),
        bias=get_tensor_from_dict(weights_dict, "final_layernorm.bias"),
    )
    model_config.final_layernorm.layernorm_type = (
        LAYERNORM_RMS if isinstance(llm_model_config, LlamaConfig) else LAYERNORM_DEFAULT
    )

    tensor_parallel_size = nemo_model_config.tensor_model_parallel_size
    pipeline_parallel_size = 1
    world_size = tensor_parallel_size*pipeline_parallel_size

    #hack since tensorrt_llm doesnt support DP natively so init all ranks with DP=1  
    model_config.mapping = tensorrt_llm.Mapping(
        world_size=tensor_parallel_size*pipeline_parallel_size, 
        rank=tensorrt_llm.mpi_rank() % world_size,
        tp_size=tensor_parallel_size, 
        pp_size=pipeline_parallel_size)
    model_config.mapping.rank = tensorrt_llm.mpi_rank()
    model_config.mapping.tp_group = get_tensor_parallel_group(tensor_parallel_size)

    LOGGER.info(f'''Resharing: Rank {tensorrt_llm.mpi_rank()} mapping:
        tp_rank  {parallel_state.get_tensor_model_parallel_rank()} -> {model_config.mapping.tp_rank}, 
        pp_rank  {parallel_state.get_pipeline_model_parallel_rank()} -> {model_config.mapping.pp_rank}, 
        tp_group {model_config.mapping.tp_group}'''
    )

    for i in range(llm_model_config.n_layer):
        model_config.layers.append(
            DecoderLayerConfig.from_nemo(
                weights_dict=weights_dict,
                llm_config=llm_model_config,
                decoder_type=decoder_type,
                layer_id=i,
                rank=model_config.mapping.tp_rank,
                is_mcore=llm_model_config.is_mcore,
            )
        )
    lm_head_weight = get_tensor_from_dict(weights_dict, "lm_head.weight")

    assert model_config.vocab_size_padded == model_config.vocab_size

    model_config.lm_head = LinearConfig(linear_type=LINEAR_COLUMN)
    model_config.lm_head.weight = lm_head_weight
    
    return [model_config]