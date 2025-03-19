#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import re
import sys
import json
import pickle
import argparse
import pandas as pd
from nltk.tokenize import sent_tokenize

import torch
import torch as t
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
)
import transformers
from huggingface_hub import login
from openai import OpenAI

sys.path.append("../")


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
overall_instruction = """\
You are a helpful assistant.
"""


def test_instruction_following_loose_ver(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp["instruction_id_list"]
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[
            instruction_id
        ]  # // import this from IFEval codebase
        instruction = instruction_cls(instruction_id)

        if isinstance(inp["kwargs"], list):
            instruction.build_description(**inp["kwargs"][index])
        else:
            instruction.build_description(**inp["kwargs"])

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp["prompt"])

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return all(is_following_list)


#
# Datamodule
#
class DataModuleActIfeval:
    def __init__(
        self,
        ifeval_eval_path,
        target_sub_inst_list,
        layer=13,
        target_token="last",
        center=True,
        scale=False,
    ):
        self.layer = layer

        ifeval_eval_df = self.load_response_df(
            os.path.join(ifeval_eval_path, "eval_results_loose.jsonl")
        )

        # // Select index
        ind = []
        for i in range(len(ifeval_eval_df)):
            if ifeval_eval_df.iloc[i]["instruction_id_list"][0] in target_sub_inst_list:
                ind.append(i)

        # // Load acts and labels
        self.labels = torch.tensor(ifeval_eval_df["follow_all_instructions"])[ind]
        self.labels = self.labels.float()
        self.acts = self.collect_acts(
            ifeval_eval_path,
            layer=self.layer,
            target_token=target_token,
            device="cuda",
            center=center,
            scale=scale,
            index_list=ind,
        )
        self.acts = self.acts.float()
        self.data = {}
        self.data["all"] = self.acts, self.labels

    def load_response_df(self, task_path, type="loose"):
        response_df = pd.DataFrame(self.readjsonl(task_path))
        return response_df

    def readjsonl(self, datapath):
        res = []
        with open(datapath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                res.append(json.loads(line))
        return res

    def load_pickle(self, filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def collect_acts(
        self,
        task_path,
        layer=13,
        target_token="last",
        device="cuda",
        center=True,
        scale=False,
        index_list=None,
    ):
        """
        Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
        First token: [1, len_input, hidden_emb]
        Last token: [1, 1, hidden_emb]
        """
        act_path = os.path.join(task_path, "activations")
        _num_act = len(os.listdir(act_path))
        acts = []
        print("num_act: ", _num_act)
        for _idx in range(_num_act):
            if index_list is not None and _idx in index_list:
                act_file_name = os.path.join(act_path, f"sample_{_idx}.pkl")
                act = self.load_pickle(act_file_name)
                self.saved_layers = act[f"output_token_{target_token}"].keys()
                act = act[f"output_token_{target_token}"][f"layer_{layer}"]
                act = act[
                    :, -1
                ]  # <-- last of the first token, no problem for last token --> [1, hidden_emb]
                acts.append(act)
        acts = torch.cat(acts, dim=0).to(device)
        if center:
            acts = acts - torch.mean(acts, dim=0)
        if scale:
            acts = acts / torch.std(acts, dim=0)
        return acts


#
# Classifier
#
class LRProbe(t.nn.Module):
    def __init__(self, d_in, binary_threshold=0.5, **kwargs):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=False), t.nn.Sigmoid())
        self.binary_threshold = binary_threshold

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def pred(self, x, iid=None, binary_threshold=None):
        binary_threshold = (
            binary_threshold if binary_threshold is not None else self.binary_threshold
        )
        return (self(x) > binary_threshold).float()

    def probability(self, x, iid=None):
        return self(x)

    def from_data(
        acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device="cpu", **kwargs
    ):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)

        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = t.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()

        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]


#
# Train a classifier
#
def train_cls(
    data_path,
    MODEL=None,
    TOKEN="first",
    LAYER=26,
    select_inst_list=None,
):
    MODEL = MODEL.split("/")[1]
    task_path_ifeval = f"{data_path}{MODEL}/ifeval_simple"

    # // Split train test
    dm = DataModuleActIfeval(
        task_path_ifeval,
        select_inst_list,
        layer=LAYER,
        target_token=TOKEN,
        center=True,
        scale=True,
    )

    train_acts, train_labels = dm.data["all"]

    # // Train probe
    probe = LRProbe.from_data(
        train_acts, train_labels, device="cuda", epochs=1000, binary_threshold=0.5
    )

    direc = probe.direction
    direc = direc / direc.norm()
    following_acts, non_following_acts = acts[labels == 1], acts[labels == 0]
    following_mean, non_following_mean = following_acts.mean(0), non_following_acts.mean(0)
    diff = (following_mean - non_following_mean) @ direc
    direc = diff * direc
    direc = direc.cpu()


#
# Entry function
#
def generate(q, model, generation_config, layer=None, direction=None, alpha=0):
    # // Prepare prompt
    prompt = format_prompt(q)
    print("=" * 20)
    print("Prompt:")
    print(prompt)
    print("=" * 20)

    # // Make sure everything is clean going in
    for module in model.model.layers:
        module._forward_hooks.clear()

    # // Tokenizer -> optional: padding=False, truncation=False, add_special_tokens=False
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    if torch.cuda.is_available():
        attention_mask = input_ids["attention_mask"].cuda()
        input_ids = input_ids["input_ids"].cuda()

    # // Generate
    direction = direction.to(model.device)
    with torch.inference_mode():
        # intervened prob
        def hook(module, input, output):
            token = 0  # // <-- targeting first token (input embedding)
            output[0][token] += (
                direction.to(output[0].device) * alpha
            )  # // <-- output is tuple (tensor)
            return output

        handle = model.model.layers[layer - 1].register_forward_hook(hook)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        handle.remove()

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
# // Calling model
def oracle_call(conversation: dict, api_key: str):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=conversation,
        max_tokens=50,
        temperature=0,
    )
    return response.choices[0].message.content


def get_prompts(prompt):
    conversation = [
        {"role": "user", "content": prompt},
    ]
    return conversation


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
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--hf_use_auth_token", type=str, default=None)
    parser.add_argument("--alpha_increase", type=float, default=0.3)
    parser.add_argument("--output_file_name", type=str, default=None)
    parser.add_argument("--direction_type", type=str, default="normal")

    args = parser.parse_args()

    # // Set data path
    data_file = os.path.join(args.data_path, "ifeval_simple.jsonl")
    model_name = args.model_name_hf.split("/")[1]
    task_path_ifeval = f"{args.data_path}/{model_name}/ifeval_simple"

    datas = readjsonl(data_file)
    ifeval_evals = readjsonl(os.path.join(task_path_ifeval, "eval_results_loose.jsonl"))
    ifeval_eval_df = pd.DataFrame(ifeval_evals)

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

    # // Get api keys
    with open("./configs/openai_api.json") as f:
        config = json.load(f)
    api_key = config["api_key"]

    # // Get prompt(gpt4 eval) -- 0:false, 1:true
    eval_prompt = """You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to score a given output for the given instruction.
        Score the output for the given instruction. The output is generated by an AI chatbot.
        You should give an overall score (an integer) on a scale of 0 to 9, where a higher score indicates better overall performance.
        
        \nDo NOT provide any explanation for your evaluation.
        Your response should be ONLY the score, an integer between 0 and 9.
        
        \n# Instruction:
        {input}
        
        \n# Output:
        {output}
        
        \n# Score of the Output (Your response should be ONLY the score, an integer between 0 and 9): """

    # // Set generate configs
    generation_config = GenerationConfig(
        do_sample=False,
        temperature=1,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=False,
        output_logits=False,
        repetition_penalty=1.0,
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
    )

    # // Set layer, alpha_increase
    layer = args.layer
    alpha_increase = args.alpha_increase

    # // Get all detailed instructions
    instruction_id_list = ifeval_eval_df["instruction_id_list"]
    label = ifeval_eval_df["follow_instruction_list"]
    inst_list = []
    for insts in instruction_id_list:
        for inst in insts:
            if inst not in inst_list:
                inst_list.append(inst)

    # // Train a classifier
    select_inst_list = inst_list
    probe, direction = train_cls(
        data_path=args.data_path,
        MODEL=args.model_name_hf,
        TOKEN="first",
        LAYER=layer,
        select_inst_list=select_inst_list,
    )
    # // Select direction type
    if args.direction_type == "normal":
        direction = direction
    elif args.direction_type == "random":
        print("This is random......", "*" * 30)
        direction = torch.randn(direction.size()).to(direction.device)

    # // Save file name
    res_path = os.path.join(args.data_path, args.model_name_hf.split("/")[-1])
    os.makedirs(res_path, exist_ok=True)
    response_file_name = os.path.join(
        res_path,
        f"RE_layer_{layer}_cls_allinst_direction_{args.direction_type}_{args.direction_type}_alpha_each_{alpha_increase}_{args.output_file_name}.jsonl",
    )
    output_file = open(response_file_name, "a", encoding="utf-8")

    for i, (data, ifeval_eval) in enumerate(zip(datas, ifeval_evals)):
        prompt = data["prompt"]
        kwargs = data["kwargs"]
        task_prompt = sent_tokenize(prompt)[0]
        inst_prompt = sent_tokenize(prompt)[-1]
        ori_response = ifeval_eval["response"]
        alpha = 0
        alpha_list = []
        response_list = []
        gpt4_quality_eval_list = []
        gpt4_inst_eval_list = []
        prob_eval_list = []
        predefined_eval_list = []
        bert_f1_list = []

        # // Try alphas
        for _ in range(4):
            alpha_list.append(alpha)
            response, act_dict = generate(
                prompt,
                model,
                generation_config,
                layer=layer,
                alpha=alpha,
                direction=direction.clone(),
            )
            response_list.append(response)
            acti = (act_dict["output_token_first"][f"layer_{layer}"][:, -1]).clone()

            # // Pre-defined eval
            follow_instructions = test_instruction_following_loose_ver(data, response)
            predefined_eval_list.append(follow_instructions)

            # // Probe eval
            prob_follow_probability = probe.probability(acti).item()
            prob_follow_instructions = probe.pred(acti, binary_threshold=0.2).item()
            prob_eval_list.append(prob_follow_probability)

            # // GPT4 eval - quality
            q = eval_prompt.format(
                input=task_prompt,
                output=response,
            )
            gpt4_quality_eval = oracle_call(get_prompts(q), api_key)
            try:
                gpt4_quality_eval = [
                    int(s) for s in re.findall(r"\d+", gpt4_quality_eval)
                ][0]
            except:
                gpt4_quality_eval = None
            gpt4_quality_eval_list.append(gpt4_quality_eval)

            # // GPT4 eval - instruction
            q = eval_prompt.format(
                input=inst_prompt,
                output=response,
            )
            gpt4_follow_instructions = oracle_call(get_prompts(q), api_key)
            try:
                gpt4_follow_instructions = [
                    int(s) for s in re.findall(r"\d+", gpt4_follow_instructions)
                ][0]
            except:
                gpt4_follow_instructions = None
            gpt4_inst_eval_list.append(gpt4_follow_instructions)

            # // Increase alpha
            alpha += alpha_increase

        data["output_list"] = response_list
        data["alpha_list"] = alpha_list
        data["predefined_follow_instructions"] = predefined_eval_list
        data["gpt4_quality_evaluation_scores"] = gpt4_quality_eval_list
        data["gpt4_instruction_evaluation_scores"] = gpt4_inst_eval_list
        data["prob_follow_instructions"] = prob_eval_list

        output_file.write(json.dumps(data, ensure_ascii=False) + "\n")

    output_file.close()
