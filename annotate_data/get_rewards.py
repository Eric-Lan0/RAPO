import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="iter2_K64.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="iter2_K64_Mreward.json",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/reward_model/ArmoRM-Llama3-8B-v0.1",  # "sfairXC/FsfairX-LLaMA3-RM-v0.1",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of responses per prompt"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}
reward_model = script_args.reward_name_or_path
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)
rm_pipe = pipeline(
    "text-classification",  # "sentiment-analysis",
    model=reward_model,
    device=device,  # tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True,
)


ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")

local_rank = Accelerator().local_process_index

data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

# print("ds[0]", ds[0])

"""
We process the data format here and query the reward model to get the rewards.
"""
def get_reward(test_texts):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    # rewards = [output[0]["score"] for output in pipe_outputs]  # sfairXC/FsfairX-LLaMA3-RM-v0.1
    # for output in pipe_outputs:
    #     print(output)
    #     print(output[0])
    reward = [output[0]["score"] for output in pipe_outputs]  # ArmoRM-Llama3-8B-v0.1
    # print("reward:", reward)
    if len(reward) == 1:
        reward = reward[0]
    return reward


def change_of_format(prom, resp):
    # To be modified according to the reward model and the LLM you use
    # Be careful about multi-turn conversions
    """
    prom = prom.replace("<s>GPT4 Correct User: ", "").replace("<|end_of_turn|>GPT4 Correct Assistant:", "")

    final_resp = resp.split("GPT4 Correct User")[0]
    """
    # prom = prom
    # final_resp = resp
    prom = prom.replace("<|start_header_id|>user<|end_header_id|>\n", "").replace("<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n", "")
    prom = prom.replace("<|start_header_id|>user<|end_header_id|>", "").replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", "")
    prom = prom.replace("<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>", "").replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", "")

    final_resp = resp

    message = [
        {"role": "user", "content": prom},
        {"role": "assistant", "content": final_resp},
    ]
    result = rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")

    return result


data = []

# tqdm is used to show the progress bar
"""
<EXAMPLE> with one prompt and one response.
<|start_header_id|>user<|end_header_id|>
What is the mathematical principle behind the order of operations, and why is it crucial in solving expressions like `(25 * 5) - (16 / 2)`?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The mathematical principle behind the order of operations is to establish standard rules for calculations to avoid ambiguity. This principle is known as BIDMAS or PEMDAS, an acronym that stands for Brackets/parentheses, Indices/Exponents, Division and Multiplication (from left to right), Addition and Subtraction (from left to right).

Without a standard order of operations, mathematical expressions could have multiple interpretations. For example, in the expression `2 + 3 * 4`, if we didn't follow the order of operations and performed the operations from left to right, we would first add 2 and 3 to get 5, then multiply by 4 to get 20. However, according to the order of operations, we should first multiply 3 and 4 to get 12, and then add 2 to get 14. These are two very different results from the same expression.

In the expression `(25 * 5) - (16 / 2)`, the order of operations tells us to perform the multiplication and division before the subtraction. If we didn't follow the order of operations and instead performed the operations from left to right, we would first multiply 25 and 5 to get 125, then subtract 16 to get 109, and finally divide by 2 to get 54.5. This is a very different result from the correct one, which is 117. 

So, the order of operations is crucial in solving mathematical expressions to ensure consistency and eliminate ambiguity.<|eot_id|>
"""
r_min, r_max = 0, 0
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        if len(sample["responses"]) < script_args.K:
            continue
        # ArmoRM-Llama3-8B-v0.1
        rewards = []

        # This is sequantial. Need to be optimized for efficiency.
        for response in sample["responses"]:
            test_texts = change_of_format(sample['prompt'], response)

            reward = get_reward(test_texts)
            reward = (reward + 4) / 6  # roughly linear transform to [0, 1]
            r_min = np.min([r_min, reward])
            r_max = np.max([r_max, reward])
            rewards.append(reward)
        # print("rewards_main:", rewards)
        data.append({"prompt": sample["prompt"], "responses": sample["responses"], "rewards": rewards})

print("r_min:", r_min)
print("r_max:", r_max)
# r_min: -0.80078125,   0.0,        -1.0546875, 0.0,    -1.390625,  -1.640625,  0.0,        0.0
# r_max: 2.234375,      2.859375,   2.6875,     5.875,  2.421875,   2.125,      5.71875,    5.46875
# normalize: r = (r + 2) / 5

# Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list = [{}] * world_size

data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []

for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)

all_rewards = [sample["rewards"] for sample in gathered_data]
# print("all_rewards[0]:", all_rewards[0])

top1_scores = np.mean(np.max(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)

if local_rank == 0:
    print(
        "Collect {} data from {} inputs. mean score {} top1 score: {}".format(
            len(gathered_data), data_size, mean_scores, top1_scores
        )
    )
    if len(gathered_data) < data_size:
        print(
            "Some of the prompts are with responses < {}. This can happen because the prompt is too long and is ignored by VLLM".format(
                script_args.K
            )
        )
    output_eval_dataset = {}
    output_eval_dataset["type"] = "text_only"
    output_eval_dataset["instances"] = gathered_data
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

    if script_args.record_dir is not None:
        with open(script_args.record_dir, "a") as f:
            f.write(str(mean_scores) + "\t" + str(top1_scores) + "\n")

