#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

python run/representation_engineering.py \
    --data_path=data/ \
    --model_name_hf=meta-llama/Llama-2-7b-chat-hf \
    --layer=14 \
    --direction_type=normal \
    --hf_use_auth_token PERSONAL_TOKEN 
