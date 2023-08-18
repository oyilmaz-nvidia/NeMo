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

import urllib.request as req
from pathlib import Path

import pytest

from nemo.export import TensorRTLLM


class TestNemoExport:

    @pytest.mark.skip()
    @pytest.mark.unit
    def test_trt_llm_export(self):
        """Here we test the trt-llm transfer and infer function"""

        self._prep_test_data()

        test_at_least_one = False
        no_error = True

        for model_name, model_info in self.test_data.items():
            if model_info["location"] == "HF":
                self._download_nemo_checkpoint(
                    model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"]
                )

            if Path(model_info["checkpoint"]).exists():
                try:
                    Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)
                    trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
                    trt_llm_exporter.export(nemo_checkpoint_path=model_info["checkpoint"], n_gpus=1)
                    output = trt_llm_exporter.forward(["test1", "how about test 2"])
                    print("output 1: ", output)
                except:
                    print("Error in TensorRT LLM.")
                    no_error = False

                try:
                    trt_llm_exporter2 = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"], gpu_id=1)
                    output = trt_llm_exporter2.forward(["Let's see how this works", "Did you get the result yet?"])
                    print("output 2: ", output)
                except:
                    print("Inference on a different GPU didn't work.")

                test_at_least_one = True

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."
        assert no_error, "At least one model couldn't be served successfully."

    @pytest.mark.unit
    def test_trt_llm_export_ptuned(self):
        """Here we test the trt-llm transfer and infer function"""

        self._prep_test_data()

        test_at_least_one = False
        no_error = True

        for model_name, model_info in self.test_data.items():
            if model_info["location"] == "HF":
                self._download_nemo_checkpoint(
                    model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"]
                )

            if Path(model_info["checkpoint"]).exists():
                #try:
                if "ptuned" in model_info:
                    for task, path in model_info["ptuned"].items():
                        print("Task: {0} and path: {1}".format(task, path))
                        Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)
                        trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
                        trt_llm_exporter.export(nemo_checkpoint_path=model_info["checkpoint"], prompt_checkpoint_path=path, n_gpus=1)
                        #output = trt_llm_exporter.forward(["test1", "how about test 2"])
                        #print("output 1: ", output)
                #except:
                 #   print("Error in TensorRT LLM.")
                 #   no_error = False

                test_at_least_one = True

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."
        assert no_error, "At least one model couldn't be served successfully."

    def _prep_test_data(self):
        self.test_data = {}

        self.test_data["GPT-843M-base"] = {}
        self.test_data["GPT-843M-base"]["location"] = "Selene"
        self.test_data["GPT-843M-base"]["trt_llm_model_dir"] = "/tmp/GPT-843M-base/trt_llm_model/"
        self.test_data["GPT-843M-base"][
            "checkpoint"
        ] = "/opt/checkpoints/GPT-843M-base/megatron_converted_843m_tp1_pp1.nemo"

        self.test_data["GPT-2B-HF-base"] = {}
        self.test_data["GPT-2B-HF-base"]["location"] = "Selene"
        self.test_data["GPT-2B-HF-base"]["trt_llm_model_dir"] = "/tmp/GPT-2B-hf-base/trt_llm_model/"
        self.test_data["GPT-2B-HF-base"]["checkpoint_dir"] = "/tmp/GPT-2B-hf-base/nemo_checkpoint/"
        self.test_data["GPT-2B-HF-base"]["checkpoint"] = (
            "/opt/checkpoints/GPT-2B.nemo"
        )
        self.test_data["GPT-2B-HF-base"]["checkpoint_link"] = (
            "https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo"
        )
        self.test_data["GPT-2B-HF-base"]["ptuned"] = {}
        self.test_data["GPT-2B-HF-base"]["ptuned"]["squad"] = "/opt/checkpoints/8b_squad_megatron_gpt_peft_tuning.nemo"

        self.test_data["GPT-2B-base"] = {}
        self.test_data["GPT-2B-base"]["location"] = "Selene"
        self.test_data["GPT-2B-base"]["trt_llm_model_dir"] = "/tmp/GPT-2B-base/trt_llm_model/"
        self.test_data["GPT-2B-base"]["checkpoint"] = "/opt/checkpoints/GPT-2B-base/megatron_converted_2b_tp1_pp1.nemo"

        self.test_data["GPT-8B-base"] = {}
        self.test_data["GPT-8B-base"]["location"] = "Selene"
        self.test_data["GPT-8B-base"]["trt_llm_model_dir"] = "/tmp/GPT-8B-base/trt_llm_model/"
        self.test_data["GPT-8B-base"]["checkpoint"] = "/opt/checkpoints/GPT-8B-base/megatron_converted_8b_tp4_pp1.nemo"

        self.test_data["GPT-43B-base"] = {}
        self.test_data["GPT-43B-base"]["location"] = "Selene"
        self.test_data["GPT-43B-base"]["trt_llm_model_dir"] = "/tmp/GPT-43B-base/trt_llm_model/"
        self.test_data["GPT-43B-base"][
            "checkpoint"
        ] = "/opt/checkpoints/GPT-43B-base/megatron_converted_43b_tp8_pp1.nemo"

    def _download_nemo_checkpoint(self, checkpoint_link, checkpoint_dir, checkpoint_path):
        if not Path(checkpoint_path).exists():
            print("Checkpoint: {0}, will be downloaded to {1}".format(checkpoint_link, checkpoint_path))
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            req.urlretrieve(checkpoint_link, checkpoint_path)
            print("Checkpoint: {0}, download completed.".format(checkpoint_link))
        else:
            print("Checkpoint: {0}, has already been downloaded.".format(checkpoint_link))
