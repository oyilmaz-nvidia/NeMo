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
import pytest
from nemo.export import TensorRTLLM
from tests.infer_data_path import get_infer_test_data, download_nemo_checkpoint
import torch
import shutil
import json


def get_accuracy_with_lambada(model):
        # lambada dataset based accuracy test, which includes more than 5000 sentences.
        # Use generated last token with original text's last token for accuracy comparison.
        # If the generated last token start with the original token, trtllm_correct make an increment.
        # It generates a CSV file for text comparison detail.

        trtllm_correct = 0
        trtllm_correct_relaxed = 0
        all_expected_outputs = []
        all_trtllm_outputs = []

        with open('/opt/NeMo/tests/deploy/lambada.json', 'r') as file:
            records = json.load(file)

            for record in records:
                prompt = record["text_before_last_word"]
                expected_output = record["last_word"].strip().lower()
                trtllm_output = model.forward(input_texts=[prompt], max_output_token=1, top_k=1, top_p=0, temperature=0.1)
                trtllm_output = trtllm_output[0][0].strip().lower()

                all_expected_outputs.append(expected_output)
                all_trtllm_outputs.append(trtllm_output)

                if expected_output == trtllm_output:
                    trtllm_correct += 1

                if expected_output == trtllm_output or trtllm_output.startswith(expected_output) or expected_output.startswith(trtllm_output):
                    if len(trtllm_output) == 1 and len(expected_output) > 1:
                        continue
                    trtllm_correct_relaxed += 1

                # print("-- expected_output: {0} and trtllm_output: {1}".format(expected_output, trtllm_output))
                
        trtllm_accuracy = trtllm_correct / len(all_expected_outputs)
        trtllm_accuracy_relaxed = trtllm_correct_relaxed / len(all_expected_outputs)
        return trtllm_accuracy, trtllm_accuracy_relaxed, all_trtllm_outputs, all_expected_outputs


def run_trt_llm_export(model_name, n_gpu, ptuning=False, tp_size=None, pp_size=None):
    test_data = get_infer_test_data()
    
    model_info = test_data[model_name]
    if model_info["location"] == "HF":
        download_nemo_checkpoint(
            model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"]
        )

    print(
        "Path: {0} and model: {1} is next and test will start if the nemo checkpoint exists ...".format(
            model_info["checkpoint"], model_name
        )
    )
    if Path(model_info["checkpoint"]).exists():
        if n_gpu > torch.cuda.device_count():
            print(
                "Path: {0} and model: {1} with {2} gpus won't be tested since available # of gpus = {3}".format(
                    model_info["checkpoint"], model_name, n_gpu, torch.cuda.device_count()
                )
            )
            return

        Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)

        print("")
        print("")
        print("################################################## NEW TEST ##################################################")
        print("")

        print(
            "Path: {0} and model: {1} with {2} gpus will be tested".format(
                model_info["checkpoint"], model_name, n_gpu
            )
        )

        prompt_embeddings_checkpoint_path = None
        if ptuning:
            if "p_tuning_checkpoint" in model_info.keys():
                prompt_embeddings_checkpoint_path=model_info["p_tuning_checkpoint"]    
                print("---- PTuning enabled.")
            else:
                print("---- PTuning could not be enabled.")

        trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])        
        trt_llm_exporter.export(
            nemo_checkpoint_path=model_info["checkpoint"],
            model_type=model_info["model_type"],
            n_gpus=n_gpu,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            max_input_token=1024,
            max_output_token=128,
            max_batch_size=model_info["max_batch_size"],
            prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path,
        )
        output = trt_llm_exporter.forward(
            input_texts=model_info["prompt_template"],
            max_output_token=model_info["max_output_token"],
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )

        print("")
        print("--- Prompt: ", model_info["prompt_template"])
        print("")
        print("--- Output: ", output)
        print("")
        print("Start model accuracy testing ...")
        trtllm_accuracy, trtllm_accuracy_relaxed, all_trtllm_outputs, all_expected_outputs = get_accuracy_with_lambada(trt_llm_exporter)
        print("Model Accuracy: {0}, Relaxed Model Accuracy: {1}".format(trtllm_accuracy, trtllm_accuracy_relaxed))
        assert trtllm_accuracy_relaxed > 0.5, "Model accuracy is below 0.5"

        trt_llm_exporter = None
        shutil.rmtree(model_info["trt_llm_model_dir"])


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Base_4k_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""

    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Base-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Base_4k_ptuning_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""

    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Base-4k", n_gpus, ptuning=True)


@pytest.mark.parametrize("n_gpus", [2])
def test_NV_GPT_8B_Base_4k_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""

    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Base-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_NV_GPT_8B_Base_4k_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""

    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Base-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_NV_GPT_8B_Base_4k_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""

    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Base-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_QA_4k_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-QA-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [2])
def test_NV_GPT_8B_QA_4k_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-QA-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_NV_GPT_8B_QA_4k_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-QA-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_NV_GPT_8B_QA_4k_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-QA-4k", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Chat_4k_SFT_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SFT", n_gpus)


@pytest.mark.parametrize("n_gpus", [2])
def test_NV_GPT_8B_Chat_4k_SFT_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SFT", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_NV_GPT_8B_Chat_4k_SFT_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SFT", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_NV_GPT_8B_Chat_4k_SFT_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SFT", n_gpus)
        

@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Chat_4k_RLHF_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-RLHF", n_gpus)


@pytest.mark.parametrize("n_gpus", [2])
def test_NV_GPT_8B_Chat_4k_RLHF_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-RLHF", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_NV_GPT_8B_Chat_4k_RLHF_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-RLHF", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_NV_GPT_8B_Chat_4k_RLHF_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-RLHF", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_NV_GPT_8B_Chat_4k_SteerLM_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SteerLM", n_gpus)


@pytest.mark.parametrize("n_gpus", [2])
def test_NV_GPT_8B_Chat_4k_SteerLM_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SteerLM", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_NV_GPT_8B_Chat_4k_SteerLM_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SteerLM", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_NV_GPT_8B_Chat_4k_SteerLM_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("NV-GPT-8B-Chat-4k-SteerLM", n_gpus)


@pytest.mark.parametrize("n_gpus", [2])
def test_GPT_43B_Base_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("GPT-43B-Base", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_GPT_43B_Base_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("GPT-43B-Base", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_GPT_43B_Base_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("GPT-43B-Base", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_LLAMA2_7B_base_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_LLAMA2_7B_base_ptuning_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus, ptuning=True)


@pytest.mark.parametrize("n_gpus", [2])
def test_LLAMA2_7B_base_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_LLAMA2_7B_base_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_LLAMA2_7B_base_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_LLAMA2_13B_base_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-13B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [1])
def test_LLAMA2_13B_base_ptuning_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-13B-base", n_gpus, ptuning=True)


@pytest.mark.parametrize("n_gpus", [2])
def test_LLAMA2_13B_base_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-13B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_LLAMA2_13B_base_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-13B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_LLAMA2_13B_base_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-13B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [2])
def test_LLAMA2_70B_base_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-70B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_LLAMA2_70B_base_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-70B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_LLAMA2_70B_base_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-70B-base", n_gpus)
    
@pytest.mark.parametrize("n_gpus", [2])
def test_LLAMA2_7B_base_tp1pp2(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus, tp_size=1, pp_size=2)

@pytest.mark.parametrize("n_gpus", [4])
def test_LLAMA2_7B_base_tp2pp2(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus, tp_size=2, pp_size=2)

@pytest.mark.parametrize("n_gpus", [8])
def test_LLAMA2_7B_base_tp1pp8(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("LLAMA2-7B-base", n_gpus, tp_size=1, pp_size=8)

@pytest.mark.parametrize("n_gpus", [1])
def test_FALCON_7B_base_1gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("FALCON-7B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [2])
def test_FALCON_40B_base_2gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("FALCON-40B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_FALCON_40B_base_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("FALCON-40B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_FALCON_40B_base_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("FALCON-40B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [4])
def test_FALCON_180B_base_4gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("FALCON-180B-base", n_gpus)


@pytest.mark.parametrize("n_gpus", [8])
def test_FALCON_180B_base_8gpu(n_gpus):
    """Here we test the trt-llm transfer and infer function"""
    if n_gpus > torch.cuda.device_count():
        pytest.skip("Skipping the test due to not enough number of GPUs", allow_module_level=True)

    run_trt_llm_export("FALCON-180B-base", n_gpus)
