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

import logging
import typing
from abc import ABC, abstractmethod

import numpy as np
from pytriton.client import ModelClient
from .tensorrt_llm_backend.client import HttpTritonClient
from .utils import str_list2numpy
import concurrent.futures

class NemoQueryBase(ABC):
    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name

    def query_llm(
            self,
            prompts,
            max_output_token=512,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            init_timeout=600.0,
    ):
        pass

class NemoQuery(NemoQueryBase):
    """
    Sends a query to Triton for LLM inference

    Example:
        from nemo.deploy import NemoQuery

        nq = NemoQuery(url="localhost", model_name="GPT-2B")

        prompts = ["hello, testing GPT inference", "another GPT inference test?"]
        output = nq.query_llm(
            prompts=prompts,
            max_output_len=100,
            top_k=1,
            top_p=0.0,
            temperature=0.0,
        )
        print("prompts: ", prompts)
    """

    def __init__(self, url, model_name):
        super().__init__(
            url=url,
            model_name=model_name,
        )

    def query_llm(
            self,
            prompts,
            max_output_token=None,
            top_k=None,
            top_p=None,
            temperature=None,
            stop_words_list=None,
            bad_words_list=None,
            no_repeat_ngram_size=None,
            init_timeout=60.0,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            prompts (List(str)): list of sentences.
            max_output_token (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            init_timeout (flat): timeout for the connection.
        """

        prompts = str_list2numpy(prompts)
        inputs = {"prompts": prompts}

        if not max_output_token is None:
            inputs["max_output_token"] = np.full(prompts.shape, max_output_token, dtype=np.int_)

        if not top_k is None:
            inputs["top_k"] = np.full(prompts.shape, top_k, dtype=np.int_)

        if not top_p is None:
            inputs["top_p"] = np.full(prompts.shape, top_p, dtype=np.single)

        if not temperature is None:
            inputs["temperature"] = np.full(prompts.shape, temperature, dtype=np.single)

        if not stop_words_list is None:
            stop_words_list = np.char.encode(stop_words_list, "utf-8")
            inputs["stop_words_list"] = np.full((prompts.shape[0], len(stop_words_list)), stop_words_list)

        if not no_repeat_ngram_size is None:
            inputs["no_repeat_ngram_size"] = np.full(prompts.shape, no_repeat_ngram_size, dtype=np.single)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(**inputs)
            output_type = client.model_config.outputs[0].dtype

        if output_type == np.bytes_:
            sentences = np.char.decode(result_dict["outputs"].astype("bytes"), "utf-8")
            return sentences
        else:
            return result_dict["outputs"]



class NemoQueryTensorRTLLM(NemoQueryBase):

    def __init__(self, url, model_name):
        super().__init__(
            url=url,
            model_name=model_name,
        )
        
    def _single_query(self,
                      prompt, max_output_token=512,
                      top_k=1,
                      top_p=0.0,
                      temperature=1.0,):
        client = HttpTritonClient(self.url)
        pload = {
            'prompt': [[prompt]], 
            'tokens': max_output_token,
            'temperature':temperature,
            'top_k': top_k,
            'top_p': top_p,
            'beam_width':1,
            'repetition_penalty':1.0,
            'length_penalty':1.0
        }
        result = client.request(self.model_name, **pload)
        return result

    def query_llm(
            self,
            prompts,
            max_output_token=512,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
    ):

        results = []
        for prompt in prompts:
            result = self._single_query(prompt, max_output_token,
                                        top_k=1,
                                        top_p=0.0,
                                        temperature=1.0,)
            results.append(result)
        return results

    def query_llm_async(
            self,
            prompts,
            max_output_token=512,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            num_threads = 12
    ):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_prompt = {executor.submit(self._single_query, prompt, max_output_token,
                                             top_k,
                                             top_p,
                                             temperature): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(
                        f"Could not run inference for prompt: - {prompt}")
                    results.append(None)

                print(len(results))
        return results
            
