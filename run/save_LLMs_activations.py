#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
)
import transformers
import torch
from tqdm import tqdm
import argparse
import os
import pickle
from huggingface_hub import login


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
overall_instruction = """\
You are a helpful assistant.
"""


#
# Entry function
#
def generate(q, history, model, generation_config):
    prompt = format_prompt(q, history)
    print("=" * 20)
    print("Prompt:")
    print(prompt)
    print("=" * 20)

    # // Tokenizer -> optional: padding=False, truncation=False, add_special_tokens=False
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if torch.cuda.is_available():
        input_ids = input_ids["input_ids"].cuda()

    # // Generate
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )

    # // Change the shape of activation
    act_dict = save_activations_as_dict(outputs)

    # // Decode output generation
    s = outputs.sequences[0][input_ids.shape[1] :]
    response = tokenizer.decode(
        s, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    print("*" * 20)
    print("Response:")
    print(response)
    print("*" * 20)
    return response, act_dict


#
# Helper functions
#
def format_prompt(query, history=[], input=None):
    prompt = ""
    if len(history) == 0:
        prompt += f"<s>{B_INST} {B_SYS} {overall_instruction} {E_SYS} {query} {E_INST} "
    else:
        for i, (old_query, response) in enumerate(history):
            prompt += f"{old_query} {response}</s>"
        prompt += f"<s>{B_INST} {query} {E_INST}"
    return prompt


def readjsonl(datapath):
    res = []
    with open(datapath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res


def save_pickle(obj, filename: str):
    with open(filename, "wb") as f:
        return pickle.dump(obj, f)


def save_activations_as_dict(outputs):
    """
    # // Original shape of outputs['hidden_states']
    outputs['hidden_states'] shape: (num_output_tokens, (num_layers, [batch_size, num_input_tokens or 1, hidden_emb]))
        first token:
            (num_layers, [batch_size, num_input_tokens, hidden_emb])
        last token:
            (num_layers, [batch_size, 1, hidden_emb])


    # // Change shape and save in save_hs dict
    save_hs :  {output_token_0:
                            {layer_0: [num_input_tokens, hidden_emb],
                            layer_1: [num_input_tokens, hidden_emb], ...},
                output_token_1:
                    {layer_0: [1, hidden_emb],
                    layer_1: [1, hidden_emb], ...}
    """
    save_hs = {}

    hs = outputs["hidden_states"]
    num_output_token = len(hs)
    num_layers = len(hs[0])

    """
    Here, for saving memory, we will only save target output_tokens and layers
        output_tokens: first, middle, and last
        layers: fisrt, 13, 20, last
    """
    target_output_tokens = {
        "first": 0,
        "middle": num_output_token // 2,
        "last": num_output_token - 1,
    }

    target_layers = []
    i = num_layers - 1
    leap = num_layers // 5
    while i > 10:
        target_layers.append(i)
        i = i - leap

    save_hs = {}
    for token_idx in target_output_tokens.keys():
        save_hs[f"output_token_{token_idx}"] = {}
        for layer_idx in target_layers:
            save_hs[f"output_token_{token_idx}"][f"layer_{layer_idx}"] = hs[
                target_output_tokens[token_idx]
            ][layer_idx].detach()

    return save_hs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        default="",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--model_name_hf",
        type=str,
        required=True,
        default="",
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        default="IFEval",
        help="ifeval, counterfact, biasbios, json, pronounce",
    )
    parser.add_argument("--hf_use_auth_token", type=str, default=None)
    args = parser.parse_args()

    # // Load data
    data_file = os.path.join(args.data_path, "ifeval_simple.jsonl")
    datas = readjsonl(data_file)

    # // Load model
    login(token=args.hf_use_auth_token)
    if "Llama-2" in args.model_name_hf:
        model_config = transformers.AutoConfig.from_pretrained(args.model_name_hf)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_hf,
            device_map="auto",
            use_auth_token=args.hf_use_auth_token,
            config=model_config,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_hf,
            device_map="auto",
            use_auth_token=args.hf_use_auth_token,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_hf, use_auth_toke=args.hf_use_auth_token
    )
    model.eval()
    print(model)

    generation_config = GenerationConfig(
        do_sample=False,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        repetition_penalty=1.0,
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
    )

    # // Save prompt, response, and activations
    res_path = os.path.join(args.data_path, args.model_name_hf.split("/")[-1])
    os.makedirs(res_path, exist_ok=True)
    response_path = os.path.join(res_path, args.task_type)
    os.makedirs(response_path, exist_ok=True)
    response_file_name = os.path.join(response_path, "response.jsonl")
    output_file = open(response_file_name, "a", encoding="utf-8")

    for idx, data in enumerate(tqdm(datas)):
        prompt = data["prompt"]
        response, act_dict = generate(prompt, [], model, generation_config)
        data["output"] = response
        output_file.write(json.dumps(data, ensure_ascii=False) + "\n")

        # // Save activation
        act_path = os.path.join(res_path, args.task_type, "activations")
        os.makedirs(act_path, exist_ok=True)
        act_file_name = os.path.join(act_path, f"sample_{idx}.pkl")
        save_pickle(act_dict, act_file_name)

    output_file.close()
