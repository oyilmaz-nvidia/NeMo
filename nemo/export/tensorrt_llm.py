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

from pathlib import Path
import os
import argparse
import pprint
import shutil

import torch
from .trt_llm.nemo_utils import get_model_config, get_tokenzier, nemo_decode, nemo_to_tensorrt_llm
from .trt_llm.tensorrt_llm_run import generate, load


class TensorRTLLM:

    def __init__(self, model_dir: str):
        if not Path(model_dir).is_dir():
            raise Exception("A valid directory path should be provided.")

        self.model_dir = model_dir
        self.model = None
        self._load()

    def _load(self):
        if len(os.listdir(self.model_dir)) > 0:
            pass

    def export(self,
               nemo_checkpoint_path,
               delete_existing_files=True,
               n_gpus=1,
               max_input_len=200,
               max_output_len=200,
               max_batch_size=32,
    ):

        if delete_existing_files and len(os.listdir(self.model_dir)) > 0:
            for files in os.listdir(self.model_dir):
                path = os.path.join(self.model_dir, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)

            if len(os.listdir(self.model_dir)) > 0:
                raise Exception("Couldn't delete all files.")
        elif len(os.listdir(self.model_dir)) > 0:
            raise Exception("There are files in this folder. Try setting delete_existing_files=True.")

        self.model = None

        weights_dir, model_config, tokenizer = nemo_decode(
            nemo_checkpoint_path, self.model_dir, tensor_parallelism=n_gpus
        )

        # We can also load the model config and tokenizer from the weights_dir.
        model_config = get_model_config(weights_dir)
        nemo_to_tensorrt_llm(
            weights_dir,
            model_config,
            self.model_dir,
            gpus=n_gpus,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
        )

        self._load()

    def infer(self, sentences):
        if self.model is None:
            raise Exception("A nemo checkpoint should be exported and "
                            "TensorRT LLM should be loaded first to run inference.")
        else:
            return None