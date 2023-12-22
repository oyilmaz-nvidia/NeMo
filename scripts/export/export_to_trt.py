# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys, os
from pathlib import Path

from nemo.deploy import DeployPyTriton, NemoQuery
from nemo.export import TensorRTLLM
import logging

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


LOGGER = logging.getLogger("NeMo")


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Exports nemo models stored in nemo checkpoints to TensorRT-LLM",
    )

    parser.add_argument(
        "-nc",
        "--nemo_checkpoint",
        required=True,
        type=str,
        help="Source .nemo file"
    )

    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=True,
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder"],
        help="Type of the model. gptnext, gpt, llama, falcon, and starcoder are only supported."
             " gptnext and gpt are the same and keeping it for backward compatibility"
    )

    parser.add_argument(
        "-mr",
        "--model_repository",
        required=True,
        default=None,
        type=str,
        help="Folder for the trt-llm model files"
    )

    parser.add_argument(
        "-ng",
        "--num_gpus",
        default=1,
        type=int,
        help="Number of GPUs for the deployment"
    )

    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bf16", "fp16", "fp8", "int8"],
        default="bf16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )

    parser.add_argument(
        "-mil",
        "--max_input_len",
        default=256,
        type=int,
        help="Max input length of the model"
    )

    parser.add_argument(
        "-mol",
        "--max_output_len",
        default=256,
        type=int,
        help="Max output length of the model"
    )

    parser.add_argument(
        "-mbs",
        "--max_batch_size",
        default=8,
        type=int,
        help="Max batch size of the model"
    )

    parser.add_argument(
        "-mpet",
        "--max_prompt_embedding_table_size",
        default=None,
        type=int,
        help="Max prompt embedding table size"
    )

    parser.add_argument(
        "-uib",
        "--use_inflight_batching",
        default="False",
        type=str,
        help="Enable inflight batching for TensorRT-LLM Triton backend."
    )

    parser.add_argument(
        "-dm",
        "--debug_mode",
        default="False",
        type=str,
        help="Enable debug mode"
    )

    args = parser.parse_args(argv)
    return args


def nemo_export(argv):
    args = get_args(argv)

    if args.debug_mode == "True":
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    if args.use_inflight_batching == "True":
        args.use_inflight_batching = True
    else:
        args.use_inflight_batching = False

    if args.dtype != "bf16":
        LOGGER.error("Only bf16 is currently supported for the optimized deployment with TensorRT-LLM. "
                      "Support for the other precisions will be added in the coming releases.")
        return

    try:
        trt_llm_exporter = TensorRTLLM(model_dir=args.model_repository)

        LOGGER.info("Export to TensorRT-LLM function is called.")
        trt_llm_exporter.export(
            nemo_checkpoint_path=args.nemo_checkpoint,
            model_type=args.model_type,
            n_gpus=args.num_gpus,
            max_input_token=args.max_input_len,
            max_output_token=args.max_output_len,
            max_batch_size=args.max_batch_size,
            max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
            use_inflight_batching=args.use_inflight_batching,
        )

        LOGGER.info("Export is successful.")
    except Exception as error:
        LOGGER.error("Error message: " + str(error))


if __name__ == '__main__':
    nemo_export(sys.argv[1:])
