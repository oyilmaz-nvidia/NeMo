import argparse

import numpy as np
import torch
from pytriton.client import ModelClient

from nemo.deploy.deploy_pytriton import DeployPyTriton
from nemo.deploy.nlp import NemoTritonQueryLLMTensorRT
from nemo.deploy.nlp.megatronllm_deployable import MegatronLLMDeployable
from nemo.deploy.nlp.query_llm import NemoTritonQueryLLMPyTorch


def test_triton_deployable(args):
    megatron_deployable = MegatronLLMDeployable(args.nemo_checkpoint, args.num_gpus)

    prompts = ["What is the biggest planet in the solar system?", "What is the fastest steam locomotive in history?"]
    url = "localhost:8000"
    model_name = args.model_name
    init_timeout = 600.0

    nm = DeployPyTriton(
        model=megatron_deployable,
        triton_model_name=model_name,
        triton_model_version=1,
        max_batch_size=8,
        port=8000,
        address="0.0.0.0",
        streaming=False,
    )
    nm.deploy()
    nm.run()

    # run once with NemoTritonQueryLLMPyTorch
    nemo_triton_query = NemoTritonQueryLLMPyTorch(url, model_name)

    result_dict = nemo_triton_query.query_llm(
        prompts,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_length=args.max_output_token,
        init_timeout=init_timeout,
    )
    print("NemoTritonQueryLLMPyTriton result:")
    print(result_dict)

    # run once with ModelClient, the results should be identical
    str_ndarray = np.array(prompts)[..., np.newaxis]
    prompts = np.char.encode(str_ndarray, "utf-8")
    max_output_token = np.full(prompts.shape, args.max_output_token, dtype=np.int_)
    top_k = np.full(prompts.shape, args.top_k, dtype=np.int_)
    top_p = np.full(prompts.shape, args.top_p, dtype=np.single)
    temperature = np.full(prompts.shape, args.temperature, dtype=np.single)

    max_output_token = np.full(prompts.shape, 1, dtype=np.int_)
    min_length = np.full(prompts.shape, 0, dtype=np.int_)

    with ModelClient(url, model_name, init_timeout_s=init_timeout) as client:
        result_dict = client.infer_batch(
            prompts=prompts, max_length=max_output_token, top_k=top_k, top_p=top_p, temperature=temperature,
        )
        print("ModelClient result:")
        print(result_dict)

    # test logprobs generation
    # right now we don't support batches where output data is inconsistent in size, so submitting each prompt individually
    all_probs = np.full(prompts.shape, True, dtype=np.bool_)
    compute_logprob = np.full(prompts.shape, True, dtype=np.bool_)
    with ModelClient(url, model_name, init_timeout_s=init_timeout) as client:
        for i in range(prompts.size):
            logprob_results = client.infer_batch(
                prompts=prompts[i : i + 1],
                min_length=min_length[i : i + 1],
                max_length=max_output_token[i : i + 1],
                top_k=top_k[i : i + 1],
                top_p=top_p[i : i + 1],
                temperature=temperature[i : i + 1],
                all_probs=all_probs[i : i + 1],
                compute_logprob=compute_logprob[i : i + 1],
            )
            print(f"ModelClient logprobs results for prompt {i}:")
            print(logprob_results)

        # logprob_results = client.infer_batch(
        #     prompts=prompts,
        #     min_length=min_length,
        #     max_length=max_output_token,
        #     top_k=top_k,
        #     top_p=top_p,
        #     temperature=temperature,
        #     all_probs=all_probs,
        #     compute_logprob=compute_logprob
        # )
        # print("Logprob results:")
        # print(logprob_results)

    nm.stop()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton and benchmark the models",
    )

    parser.add_argument(
        "--model_name", type=str, required=True,
    )
    # parser.add_argument(
    #     "--existing_test_models", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--model_type", type=str, required=False,
    # )
    # parser.add_argument(
    #     "--min_gpus", type=int, default=1, required=True,
    # )
    # parser.add_argument(
    #     "--max_gpus", type=int,
    # )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
    )
    # parser.add_argument(
    #     "--checkpoint_dir", type=str, default="/tmp/nemo_checkpoint/", required=False,
    # )
    parser.add_argument(
        "--nemo_checkpoint", type=str, required=True,
    )
    # parser.add_argument(
    #     "--trt_llm_model_dir", type=str,
    # )
    parser.add_argument(
        "--max_batch_size", type=int, default=8,
    )
    # parser.add_argument(
    #     "--max_input_token", type=int, default=256,
    # )
    parser.add_argument(
        "--max_output_token", type=int, default=128,
    )
    # parser.add_argument(
    #     "--p_tuning_checkpoint", type=str,
    # )
    # parser.add_argument(
    #     "--ptuning", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--lora_checkpoint", type=str,
    # )
    # parser.add_argument(
    #     "--lora", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--tp_size", type=int,
    # )
    # parser.add_argument(
    #     "--pp_size", type=int,
    # )
    parser.add_argument(
        "--top_k", type=int, default=1,
    )
    parser.add_argument(
        "--top_p", type=float, default=0.0,
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
    )
    # parser.add_argument(
    #     "--run_accuracy", default=False, action='store_true',
    # )
    # parser.add_argument("--streaming", default=False, action="store_true")
    # parser.add_argument(
    #     "--test_deployment", type=str, default="False",
    # )
    # parser.add_argument(
    #     "--debug", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--ci_upload_test_results_to_cloud", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--test_data_path", type=str, default=None,
    # )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test_triton_deployable(args)
