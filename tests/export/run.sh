echo "unset all SLURM_, PMI_, PMIX_ Variables"
set -x
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done
set +x


python tests/export/test_nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --min_gpus 1 --max_gpus 2
python tests/export/test_nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --min_gpus 1 --streaming
python tests/export/test_nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --min_gpus 1 --max_gpus 2 --tp_size 1 --pp_size=2
python tests/export/test_nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --min_gpus 1 --max_gpus 2 --tp_size 2 --pp_size=2
python tests/export/test_nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --min_gpus 1 --max_gpus 2 --tp_size 1 --pp_size=8
python tests/export/test_nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --ptuning --min_gpus 1 --max_gpus 2
python tests/export/test_nemo_export.py --model_name LLAMA2-7B-code --existing_test_models --min_gpus 1 --max_gpus 2
python tests/export/test_nemo_export.py --model_name LLAMA2-13B-base --existing_test_models --min_gpus 1 --max_gpus 2
python tests/export/test_nemo_export.py --model_name LLAMA2-13B-base --existing_test_models --ptuning --min_gpus 1 --max_gpus 2
python tests/export/test_nemo_export.py --model_name LLAMA2-70B-base --existing_test_models --min_gpus 2 --max_gpus 8
python tests/export/test_nemo_export.py --model_name NV-GPT-8B-Base-4k --existing_test_models --min_gpus 1 --max_gpus 8
python tests/export/test_nemo_export.py --model_name NV-GPT-8B-QA-4k --existing_test_models --min_gpus 1 --max_gpus 8
python tests/export/test_nemo_export.py --model_name NV-GPT-8B-Chat-4k-SFT --existing_test_models --min_gpus 1 --max_gpus 8
python tests/export/test_nemo_export.py --model_name NV-GPT-8B-Chat-4k-RLHF --existing_test_models --min_gpus 1 --max_gpus 8
python tests/export/test_nemo_export.py --model_name NV-GPT-8B-Chat-4k-SteerLM --existing_test_models --min_gpus 1 --max_gpus 8
python tests/export/test_nemo_export.py --model_name GPT-43B-Base --existing_test_models --min_gpus 2 --max_gpus 8
python tests/export/test_nemo_export.py --model_name FALCON-7B-base --existing_test_models --min_gpus 1 --max_gpus 2
python tests/export/test_nemo_export.py --model_name FALCON-40B-base --existing_test_models --min_gpus 2 --max_gpus 8
python tests/export/test_nemo_export.py --model_name FALCON-180B-base --existing_test_models --min_gpus 8 --max_gpus 8



