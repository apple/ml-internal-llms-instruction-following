#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

python run/save_LLMs_activations.py \
    --data_path=data/ \
    --model_name_hf=microsoft/Phi-3-mini-128k-instruct \
    --task_type=ifeval_simple \
    --hf_use_auth_token PERSONAL_TOKEN

python run/save_LLMs_activations.py \
    --data_path=data/ \
    --model_name_hf=mistralai/Mistral-7B-Instruct-v0.3 \
    --task_type=ifeval_simple \
    --hf_use_auth_token PERSONAL_TOKEN

python run/save_LLMs_activations.py \
    --data_path=data/ \
    --model_name_hf=meta-llama/Llama-2-13b-chat-hf \
    --task_type=ifeval_simple \
    --hf_use_auth_token PERSONAL_TOKEN

python run/save_LLMs_activations.py \
    --data_path=data/ \
    --model_name_hf=meta-llama/Llama-2-7b-chat-hf \
    --task_type=ifeval_simple \
    --hf_use_auth_token PERSONAL_TOKEN