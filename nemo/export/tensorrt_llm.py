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

import os
import shutil
from pathlib import Path

import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import Tensor

from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import str_ndarray2list, cast_output

from .trt_llm.model_config_trt import model_config_to_tensorrt_llm
from .trt_llm.nemo_utils import nemo_to_model_config, get_tokenzier
from .trt_llm.quantization_utils import naive_quantization
from .trt_llm.tensorrt_llm_run import generate, load
from .utils import get_prompt_embedding_table, is_nemo_file, torch_to_numpy


class TensorRTLLM(ITritonDeployable):
    def __init__(self, model_dir: str, gpu_id=None):
        if not Path(model_dir).is_dir():
            raise Exception("A valid directory path should be provided.")

        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.n_gpus = None
        self.gpu_id = gpu_id
        self._load()

    def _load(self):
        self.tokenizer = None
        self.model = None

        folders = os.listdir(self.model_dir)
        if len(folders) > 0:
            try:
                self.tokenizer = get_tokenzier(Path(os.path.join(self.model_dir)))
                self.model = load(tokenizer=self.tokenizer, engine_dir=self.model_dir, gpu_id=self.gpu_id)
            except:
                raise Exception("Files in the TensorRT-LLM folder is corrupted and model needs to be exported again.")

    def extract_prompt_embeddings(self, prompt_checkpoint_path):
        if is_nemo_file(prompt_checkpoint_path):
            vtokens_embeddings = get_prompt_embedding_table(prompt_checkpoint_path)
            np.save(os.path.join(self.model_dir, "prompt_embeddings.npy"), torch_to_numpy(vtokens_embeddings))
        else:
            raise FileNotFoundError("file: {0} does not exist or is not a nemo file.".format(prompt_checkpoint_path))

    def export(
        self,
        nemo_checkpoint_path,
        prompt_checkpoint_path=None,
        delete_existing_files=True,
        n_gpus=1,
        max_input_len=200,
        max_output_len=200,
        max_prompt_embedding_table_size=100,
        max_batch_size=32,
        quantization=None,
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

        nemo_export_dir = os.path.join(self.model_dir, "/tmp_nemo/")
        model_configs, self.tokenizer = nemo_to_model_config(in_file=nemo_checkpoint_path, decoder_type="gptnext", gpus=n_gpus, nemo_export_dir=nemo_export_dir)

        for model_config in model_configs:
            if quantization is not None:
                naive_quantization(model_config, quantization)

        if prompt_checkpoint_path is None:
            max_prompt_embedding_table_size = 0

        model_config_to_tensorrt_llm(
            model_configs,
            self.model_dir,
            n_gpus,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        )

        if prompt_checkpoint_path is not None:
            self.extract_prompt_embeddings(prompt_checkpoint_path)

        shutil.copy(os.path.join(nemo_export_dir, "tokenizer.model"), self.model_dir)
        shutil.rmtree(nemo_export_dir)
        self._load()

    def forward(self, input_texts, input_len=0, max_output_len=200):
        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported and " "TensorRT LLM should be loaded first to run inference."
            )
        else:
            if input_len > 0:
                input_tokens = self.tokenizer.encode(input_texts + "\n")
                input_tokens = input_tokens * (int(input_len / len(input_tokens)) + 1)
                input_text = self.tokenizer.decode(input_tokens[: input_len])
                print(f"Overriding with dummy input: len {input_len}, text: {input_text}")
                input_texts = [input_text]
            return generate(input_texts, max_output_len, self.model)

    @property
    def get_triton_input(self):
        inputs = (Tensor(name="prompts", shape=(1,), dtype=bytes),)
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(1,), dtype=bytes),)
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        input_texts = str_ndarray2list(inputs.pop("prompts"))
        output_texts = self.forward(input_texts)
        output = cast_output(output_texts, np.bytes_)
        return {"outputs": output}

