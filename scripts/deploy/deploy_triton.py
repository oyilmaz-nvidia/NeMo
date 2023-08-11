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
from nemo.utils import logging

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton",
    )
    parser.add_argument("--nemo_checkpoint", type=str, help="Source .nemo file")
    parser.add_argument("--triton_model_name", required=True, type=str, help="Name for the service")
    parser.add_argument("--triton_model_version", default=1, type=int, help="Name for the service")
    parser.add_argument("--optimized", default="True", type=str, help="Use TRT-LLM for inference")
    parser.add_argument("--trt_llm_folder", default=None, type=str, help="Folder for the trt-llm conversion")
    parser.add_arugment("--num_gpu", default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp8", "int8"],
        default="bf16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )
    parser.add_argument("--verbose", default=False, help="Verbose level for logging, numeric")

    args = parser.parse_args(argv)
    return args


def nemo_deploy(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))
    logging.info(args)

    if args.dtype != "bf16":
        logging.error("Only bf16 is currently supported for the optimized deployment with TensorRT-LLM.")
        return

    if args.optimized == "True":
        if args.trt_llm_folder is None:
            trt_llm_path = "/tmp/trt_llm_model_dir/"
            logging.info(
                "/tmp/trt_llm_model_dir/ path will be used as the TensorRT LLM folder. "
                "Please set this parameter if you'd like to use a path that has already"
                "included the TensorRT LLM model files."
            )
            Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
        else:
            trt_llm_path = args.trt_llm_folder

        if args.nemo_checkpoint is None and not os.path.isdir(args.trt_llm_folder):
            logging.error(
                "The supplied trt_llm_folder is not already a valid TRT engine "
                "directory. Please supply a --nemo_checkpoint."
            )

        trt_llm_exporter = TensorRTLLM(model_dir=trt_llm_path)

        if args.nemo_checkpoint is not None:
            trt_llm_exporter.export(nemo_checkpoint_path=args.nemo_checkpoint, n_gpus=args.num_gpu)
        nm = DeployPyTriton(
            model=trt_llm_exporter,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
        )
    else:
        nm = DeployPyTriton(
            checkpoint_path=args.nemo_checkpoint,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
        )

    nm.deploy()

    try:
        logging.info("Triton deploy function is called.")
        nm.serve()
        logging.info("Model is being served.")
    except:
        logging.info("An error has occurred and will stop serving the model.")

    nm.stop()


if __name__ == '__main__':
    nemo_deploy(sys.argv[1:])
