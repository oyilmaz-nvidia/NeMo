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

import json
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pickle
import tensorrt_llm
import torch
import tempfile

from nemo.deploy import ITritonDeployable
try:
    from nemo.deploy.utils import cast_output, str_ndarray2list
except:
    pass
import logging

from .trt_llm.model_config_trt import model_config_to_tensorrt_llm
from .trt_llm.nemo_utils import get_tokenzier, nemo_to_model_config
from .trt_llm.tensorrt_llm_run import generate, load
from .utils import is_nemo_file, unpack_nemo_ckpt

use_pytriton = True
try:
    from pytriton.decorators import batch
    from pytriton.model_config import Tensor
except Exception:
    use_pytriton = False

LOGGER = logging.getLogger("NeMo")


class TensorRTLLM(ITritonDeployable):

    """
    Exports nemo checkpoints to TensorRT-LLM and run fast inference.

    Example:
        from nemo.export import TensorRTLLM

        trt_llm_exporter = TensorRTLLM(model_dir="/path/for/model/files")
        trt_llm_exporter.export(
            nemo_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="llama",
            n_gpus=1,
        )

        output = trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
        print("output: ", output)

    """

    def __init__(self, model_dir: str, load_model: bool=True):
        """
        Args:
            model_dir (str): path for storing the TensorRT-LLM model files.
            load_model (bool): load TensorRT-LLM model if the engine files exist in the model_dir.
        """

        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.n_gpus = None
        self.config = None
        self.ptuning_tables = []
        self.p_table = None
        self.task_vocab_size = 0
        self.task_ids = {}

        if load_model:
            self._load()

    def export(
        self,
        nemo_checkpoint_path: str,
        model_type: str,
        delete_existing_files: bool = True,
        n_gpus: int = 1,
        tensor_parallel_size = None,
        pipeline_parallel_size = None,
        max_input_token: int = 256,
        max_output_token: int = 256,
        max_batch_size: int = 8,
        max_prompt_embedding_table_size=None,
        use_inflight_batching: bool = False,
        enable_context_fmha: bool = True,
        paged_kv_cache: bool = False,
        dtype: str = "bfloat16",
        load_model: bool = True,
        enable_multi_block_mode = False,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            nemo_checkpoint_path (str): path for the nemo checkpoint.
            model_type (str): type of the model. Currently, "llama", "gptnext", "falcon", and "starcoder" are supported.
            delete_existing_files (bool): if Truen, deletes all the files in model_dir.
            n_gpus (int): number of GPUs to use for inference.
            tensor_parallel_size (int): tensor parallelism.
            pipeline_parallel_size (int): pipeline parallelism.
            max_input_token (int): max input length.
            max_output_token (int): max output length.
            max_batch_size (int): max batch size.
            max_prompt_embedding_table_size (int): max prompt embedding size.
            use_inflight_batching (bool): if True, enables inflight batching for TensorRT-LLM Triton backend.
            enable_context_fmha (bool): if True, use fused Context MultiHeadedAttention.
            paged_kv_cache (bool): if True, uses kv cache feature of the TensorRT-LLM.
            dtype (str): Floating point type for model weights (Supports BFloat16/Float16).
            load_model (bool): load TensorRT-LLM model after the export.
            enable_multi_block_mode (bool): enable faster decoding in multihead attention. Required for long context.
        """

        if not model_type in self.get_supported_models_list:
            raise Exception("Model {0} is not currently a supported model type. "
                            "Supported model types are llama, gptnext, falcon, and starcoder".format(model_type))

        if model_type == "gpt" or model_type == "starcoder":
            # gpt and gptnext are the same. Keeping the gptnext due to backward compatibility.
            # gpt and starcoder use the similar model architecture. So, gpt can be used for starcoder.
            model_type = "gptnext"

        if pipeline_parallel_size is None:
            tensor_parallel_size = n_gpus
            pipeline_parallel_size = 1
        elif tensor_parallel_size is None:
            tensor_parallel_size = 1
            pipeline_parallel_size = n_gpus

        if Path(self.model_dir).exists():
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
        else:
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        if max_prompt_embedding_table_size is None:
            max_prompt_embedding_table_size = 0

        self.model = None

        tmp_dir = tempfile.TemporaryDirectory()
        nemo_export_dir = Path(tmp_dir.name)

        model_configs, self.tokenizer = nemo_to_model_config(
            in_file=nemo_checkpoint_path,
            decoder_type=model_type,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            nemo_export_dir=nemo_export_dir,
        )

        model_config_to_tensorrt_llm(
            model_configs,
            self.model_dir,
            world_size=tensor_parallel_size*pipeline_parallel_size,
            max_input_len=max_input_token,
            max_output_len=max_output_token,
            max_batch_size=max_batch_size,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            use_inflight_batching=use_inflight_batching,
            paged_kv_cache=paged_kv_cache,
            enable_context_fmha=enable_context_fmha,
            enable_multi_block_mode=enable_multi_block_mode,
        )

        tokenizer_path = os.path.join(nemo_export_dir, "tokenizer.model")
        if os.path.exists(tokenizer_path):
            shutil.copy(tokenizer_path, self.model_dir)
        else:
            self.tokenizer.save_pretrained(os.path.join(self.model_dir, 'huggingface_tokenizer'))
        tmp_dir.cleanup()

        if load_model:
            self._load()

    def forward(
        self,
        input_texts: List[str],
        max_output_token: int = 64,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        stop_words_list: List[str] = None,
        bad_words_list: List[str] = None,
        no_repeat_ngram_size: int = None,
        task_ids: List[str] = None,
        prompt_embeddings_table = None,
        prompt_embeddings_checkpoint_path: str = None,
        streaming: bool = False,
        **sampling_kwargs,
    ):

        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            input_texts (List(str)): list of sentences.
            max_output_token (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            task_ids (List(str)): list of the task ids for the prompt tables.
            prompt_embeddings_table (List(float)): prompt embeddings table.
            prompt_embeddings_checkpoint_path (str): path for the nemo checkpoint for the prompt embedding table.
            sampling_kwargs: Additional kwargs to set in the SamplingConfig.
        """

        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported to TensorRT-LLM and "
                "then it should be loaded first to run inference."
            )
        else:
            if not prompt_embeddings_table is None or not prompt_embeddings_checkpoint_path is None:
                prompt_table = self._get_prompt_embedding_table(
                    prompt_embeddings_table, prompt_embeddings_checkpoint_path
                )
                tv_size = prompt_table.size(dim=0)
            elif len(self.ptuning_tables) > 0:
                prompt_table = self.p_table
                tv_size = self.task_vocab_size
            else:
                prompt_table = None
                tv_size = None

            if task_ids is None:
                assert prompt_table is None, "There is a prompt embedding table and task_ids cannot be None"
                input_task_ids = None
            else:
                if prompt_table is None:
                    input_task_ids = None
                else:
                    if len(task_ids) > 1:
                        assert len(task_ids) == len(input_texts), ("Either len of the task_ids has to be 1 or"
                                                                   "it needs to match with len of input_texts.")

                    if len(task_ids) == 1:
                        assert task_ids[0] in self.task_ids.keys(), "Task: {0} doesn't exist in the task list.".format(task_ids[0])
                        input_task_ids = [self.task_ids[task_ids[0]] for i in range(len(input_texts))]
                    else:
                        input_task_ids = []
                        for i in range(len(input_texts)):
                            assert task_ids[i] in self.task_ids.keys(), "Task: {0} doesn't exist in the task list.".format(task_ids[i])
                            input_task_ids.append(self.task_ids[task_ids[i]])

            return generate(
                input_texts=input_texts,
                max_output_len=max_output_token,
                host_context=self.model,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                prompt_table=prompt_table,
                task_vocab_size=tv_size,
                task_ids=input_task_ids,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streaming=streaming,
                **sampling_kwargs,
            )

    def add_prompt_table(self, task_name: str, prompt_embeddings_checkpoint_path: str):
        # TODO: check if the added table's size is larger than the max_prompt_embedding_table_size
        #       If yes, then raise an error.

        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported to TensorRT-LLM and "
                "then it should be loaded first to run inference."
            )

        for pt in self.ptuning_tables:
            if pt["task_name"] == task_name:
                raise Exception(
                    "Task name: {0} has already added. Please pass a unique task name.".format(task_name)
                )

        prompt_table = self._get_prompt_embedding_table(
            prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path
        )

        self.ptuning_tables.append({"table": prompt_table, "task_name": task_name})
        with open(os.path.join(self.model_dir, 'prompt_tables.pkl'), 'wb') as f:
            pickle.dump(self.ptuning_tables, f)

        self._prep_ptuning_table()

    def remove_prompt_table(self, task_name: str):
        if not self.ptuning_tables is None:
            for i in range(len(self.ptuning_tables)):
                if self.ptuning_tables[i]["task_name"] == task_name:
                    self.ptuning_tables.pop(i)
                    with open(os.path.join(self.model_dir, 'prompt_tables.pkl'), 'wb') as f:
                        pickle.dump(self.ptuning_tables, f)
                    return
            self._prep_ptuning_table()

    @property
    def get_supported_models_list(self):
        # gpt and gptnext are the same. Keeping the gptnext due to backward compatibility.
        return ["gpt", "gptnext", "llama", "falcon", "starcoder"]

    @property
    def get_hidden_size(self):
        if self.config is None:
            return None
        else:
            return self.config["builder_config"]["hidden_size"]

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_output_token", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="stop_words_list", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="bad_words_list", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="no_repeat_ngram_size", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="task_id", shape=(-1,), dtype=bytes, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(-1,), dtype=bytes),)
        return outputs


    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        try:
            infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
            if "max_output_token" in inputs:
                infer_input["max_output_token"] = inputs.pop("max_output_token")[0][0]
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")[0][0]
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")[0][0]
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")[0][0]
            if "random_seed" in inputs:
                infer_input["random_seed"] = inputs.pop("random_seed")[0][0]
            if "stop_words_list" in inputs:
                swl = np.char.decode(inputs.pop("stop_words_list").astype("bytes"), encoding="utf-8")
                infer_input["stop_words_list"] = swl[0]
            if "bad_words_list" in inputs:
                swl = np.char.decode(inputs.pop("bad_words_list").astype("bytes"), encoding="utf-8")
                infer_input["bad_words_list"] = swl[0]
            if "no_repeat_ngram_size" in inputs:
                infer_input["no_repeat_ngram_size"] = inputs.pop("no_repeat_ngram_size")[0][0]
            if "task_id" in inputs:
                task_id = np.char.decode(inputs.pop("task_id").astype("bytes"), encoding="utf-8")
                infer_input["task_ids"] = task_id[0]

            output_texts = self.forward(**infer_input)
            output = cast_output(output_texts, np.bytes_)
            return {"outputs": output}
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output = cast_output([err_msg], np.bytes_)
            return {"outputs": output}
    @batch
    def triton_infer_fn_streaming(self, **inputs: np.ndarray):
        try:
            infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
            if "max_output_token" in inputs:
                infer_input["max_output_token"] = inputs.pop("max_output_token")[0][0]
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")[0][0]
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")[0][0]
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")[0][0]
            if "random_seed" in inputs:
                infer_input["random_seed"] = inputs.pop("random_seed")[0][0]
            if "stop_words_list" in inputs:
                swl = np.char.decode(inputs.pop("stop_words_list").astype("bytes"), encoding="utf-8")
                infer_input["stop_words_list"] = swl[0]
            if "bad_words_list" in inputs:
                swl = np.char.decode(inputs.pop("bad_words_list").astype("bytes"), encoding="utf-8")
                infer_input["bad_words_list"] = swl[0]
            if "no_repeat_ngram_size" in inputs:
                infer_input["no_repeat_ngram_size"] = inputs.pop("no_repeat_ngram_size")[0][0]
            if "task_id" in inputs:
                task_id = np.char.decode(inputs.pop("task_id").astype("bytes"), encoding="utf-8")
                infer_input["task_ids"] = task_id[0]

            outputs = self.forward(**infer_input, streaming=True)
 
            for request_output in outputs:
                yield {"outputs": cast_output(request_output,  np.bytes_)}
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output = cast_output([err_msg], np.bytes_)
            return {"outputs": output}

    def _prep_ptuning_table(self):
        self.task_vocab_size = 0
        for pt in self.ptuning_tables:
            if self.task_vocab_size < pt["table"].size(dim=0):
                self.task_vocab_size = pt["table"].size(dim=0)

        # pad tasks to longest task embedding table
        vtokens_embeddings = []
        self.task_ids = {}
        tid = 0
        for i, ptuning_table in enumerate(self.ptuning_tables):
            padded_table = torch.zeros((self.task_vocab_size, self.get_hidden_size))
            padded_table[:ptuning_table["table"].size(dim=0), :] = ptuning_table["table"]
            vtokens_embeddings.append(padded_table)
            self.task_ids[ptuning_table["task_name"]] = tid
            tid = tid + 1

        if len(vtokens_embeddings) > 0:
            self.p_table = torch.stack(vtokens_embeddings, dim=0).view(-1, self.get_hidden_size)
        else:
            self.p_table = None

    def _load_prompt_tables(self):
        if not self.model_dir is None:
            pt_path = Path(os.path.join(self.model_dir, 'prompt_tables.pkl'))
            if pt_path.exists():
                with open(pt_path, 'rb') as f:
                    self.ptuning_tables = pickle.load(f)
                self._prep_ptuning_table()
            else:
                self.ptuning_tables = []

    def _get_prompt_embedding_table_ckpt(self, prompt_embeddings_checkpoint_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            unpack_nemo_ckpt(prompt_embeddings_checkpoint_path, temp_dir)
            mw_path = os.path.join(temp_dir, "model_weights.ckpt")
            if not Path(mw_path).exists():
                mw_path = os.path.join(temp_dir, "mp_rank_00", "model_weights.ckpt")
                if not Path(mw_path).exists():
                    raise FileNotFoundError("File: {0} could not be found in the nemo checkpoint. "
                                            "Please check the nemo checkpoint format for the prompt "
                                            "embedding table.".format(mw_path))
            weights = torch.load(mw_path)
            weights = weights["model.embedding.adapter_layer.ptuning_adapter.inference_table"]

            return weights.cpu().detach()
        return None

    def _get_prompt_embedding_table(self, prompt_embeddings_table=None, prompt_embeddings_checkpoint_path=None, ):
        p_tuning = "no_ptuning"
        if (prompt_embeddings_table is not None and
                prompt_embeddings_checkpoint_path is not None
        ):
            LOGGER.warning("prompt_embeddings_table will be used and "
                           "prompt_embeddings_checkpoint_path will be "
                           "ignored for ptuning.")
            p_tuning = "use_table"
        elif prompt_embeddings_table is not None:
            p_tuning = "use_table"
        elif prompt_embeddings_checkpoint_path is not None:
            p_tuning = "use_checkpoint"
        else:
            return None, None

        if p_tuning == "use_table":
            if not isinstance(prompt_embeddings_table, np.ndarray):
                raise TypeError("Only numpy array is allowed for the prompt embeddings table.")

            if len(prompt_embeddings_table.shape) != 2:
                raise Exception("A two dimensional prompt embeddings table for a single task is only supported.")

            prompt_embeddings_table = torch.from_numpy(prompt_embeddings_table)
        elif p_tuning == "use_checkpoint":
            if not is_nemo_file(prompt_embeddings_checkpoint_path):
                raise TypeError(prompt_embeddings_checkpoint_path + " is not a nemo file.")
            prompt_embeddings_table = self._get_prompt_embedding_table_ckpt(prompt_embeddings_checkpoint_path)

        dtype = self.config['builder_config']['precision']
        prompt_embeddings_table = prompt_embeddings_table.to(dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)).cuda()

        if prompt_embeddings_table.size(dim=1) != self.config["builder_config"]["hidden_size"]:
            raise Exception(
                "Hidden dimension of the model is {0} and does not match with the dimension of the prompt table.".format(
                    self.config["builder_config"]["hidden_size"])
            )

        return prompt_embeddings_table

    def _load_config_file(self):
        engine_dir = Path(self.model_dir)
        config_path = engine_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError("file: {0} could not be found.".format(config_path))

    def _load(self):
        self.model = None
        self.tokenizer = None
        self.n_gpus = None
        self.config = None
        self.ptuning_tables = []

        if Path(self.model_dir).exists():
            folders = os.listdir(self.model_dir)
            if len(folders) > 0:
                try:
                    self._load_config_file()
                    self.tokenizer = get_tokenzier(Path(os.path.join(self.model_dir)))
                    self.model = load(tokenizer=self.tokenizer, engine_dir=self.model_dir)
                    self._load_prompt_tables()
                except Exception as error:
                    raise Exception(
                        "Files in the TensorRT-LLM folder is corrupted and "
                        "model needs to be exported again. "
                        "Error message: " + str(error)
                    )
