import re
import gc
import json
import tokenize
from unittest import result
import torch
import warnings
import threading
import concurrent.futures
from typing import List, Dict, Any

from torch.nn.modules import padding
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.dummy_pt_objects import AutoModelForCausalLM
from deepspeed.accelerator import get_accelerator
from custom_agent.agent_dataset import prompts
from custom_agent.search_tools import google_search_arxiv_id
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from threading import Lock
import math
import numpy as np
from peft import PeftModel

# hyperparameters
MAX_PAPERS = 5
select_prompt = "You are an elite researcher in the field of AI, conducting research on {user_query}. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\nSearched Paper:\nTitle: {title}\nAbstract: {abstract}\n\nUser Query: {user_query}\n\nOutput format: Decision: True/False\nReason:... \nDecision:"

# regular expressions
search_user_query_template = r"User Query:(.*?)assistant\n\["
search_template = r"Search\](.*?)\["
expand_user_query_template = r"research on `(.*?)`\."
expand_template = r"Expand\](.*?)\["


def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = "".join(letters)
    return result.lower()


def search_paper_by_title(title, paper_db):
    title_key = keep_letters(title)
    if title_key in paper_db.namelist():
        with paper_db.open(title_key) as f:
            return json.loads(f.read().decode("utf-8"))
    else:
        return None


def get_expand_papers(section, paper, paper_db):
    section = keep_letters(section)
    res = []
    for sec in paper["sections"]:
        if keep_letters(sec) == section:
            if keep_letters(sec) == section:
                for title in paper["sections"][sec]:
                    p = search_paper_by_title(title, paper_db)
                    if p is not None:
                        res.append(p)
    return res


def gen_value_model_prompt(title, user_query, paper_db):
    paper = search_paper_by_title(title, paper_db)
    if paper in None:
        return None, None

    value_model_prompt = [
        {
            "role": "user",
            "content": prompts["select_section"]
            .format(
                user_query=user_query,
                title=title,
                abstract=paper["abstract"],
                sections=json.dumps(list(paper["sections"].keys())),
            )
            .strip(),
        },
        {
            "role": "assistant",
            "content": "[",
        },  # use the value of token '[' to approximate the value of the paper
    ]
    return value_model_prompt, paper


def call_vm(value_model_prompts, tokenizer, device, value_model):
    if len(value_model_prompts) > 0:
        input_ids = tokenizer.apply_chat_template(
            value_model_prompts,
            tokenize=True,
            padding=True,
            truncation=True,
            max_length=992,
            add_generation_paompt=False,
        )
        input_ids = torch.tensor(input_ids, device=device)
        attention_mask = input_ids != tokenizer.pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)
        output = value_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        reward_logits = value_model.score(output.hidden_states[-1])
        reward_logits = reward_logits.squeeze(-1)[:, -2].squeeze(-1)
        score = reward_logits.sum().item()
    else:
        input_ids = tokenizer.apply_chat_template(
            [[{"role": "user", "content": "hello"}]],
            tokenize=True,
            padding=True,
            padding_side="left",
            add_generation_prompt=True,
        )
        input_ids = torch.tensor(input_ids, device=device)
        attention_mask = input_ids != tokenizer.pad_token_id
        position_ids = attention_mask.cunsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)
        output = value_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        reward_logits = value_model.score(output.hidden_states[-1])
        score = 0
    del input_ids, attention_mask, position_ids, output, reward_logits
    gc.collect()
    torch.cuda.empty_cache()
    get_accelerator().empty_cache()
    return score


# llm = LLM(
#     model="/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
#     enable_lora=True,
#     dtype="bfloat16",
#     gpu_memory_utilization=0.20,
#     max_num_seqs=1,
#     max_model_len=1024,
# )
# selector_tokenizer = AutoTokenizer.from_pretrained(
#     "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
#     padding_side="left",
#     trust_remote_code=True,
# )
# base_model = AutoModelForCausalLM.from_pretrained(
#     "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
#     trust_remote_code=True,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# ).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(
#     "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
#     padding_side="left",
#     trust_remote_code=True,
# )
# select_model = PeftModel.from_pretrained(base_model, "/root/pasa/results/sft_selector/checkpoint-4957").to("cuda")


def call_selector(select_prompts: List[str]) -> List[Dict[str, Any]]:
    # warnings.warn(
    #     "Deploy an additional selector to get the relevant possibilities of paper and user_query.\nImplement a function `call_selector` that takes the prompts of the selector as input and returns the probability scores."
    # )
    # if len(select_prompts) == 0:
    #     return [{"prob": 0} for _ in range(len(select_prompts))]
    # encoded_input = tokenizer(
    #     select_prompts, return_tensors="pt", padding=True, truncation=True
    # )
    # input_ids = encoded_input.input_ids.cuda(select_model.device)
    # attention_mask = encoded_input.attention_mask.cuda(select_model.device)

    # outputs = select_model.generate(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     max_new_tokens=1,
    #     output_scores=True,
    #     return_dict_in_generate=True,
    #     do_sample=False,
    # )
    # true_token_id = tokenizer.convert_tokens_to_ids("True")
    # probs = outputs.scores[0].softmax(dim=-1)[:, true_token_id].cpu().numpy().tolist()
    # return [{"prob": p} for p in probs]

    return [{"prob": 0} for _ in range(len(select_prompts))]


def response_handler(
    num,
    response,
    all_papers,
    all_scores,
    lock,
    query_responses,
    tokenizer,
    context_length,
    value_model,
    args,
    paper_db,
    paper_id,
    typ="search",
    f_paper=None,
    answer=[],
):
    scores, has_stop = [], False
    if typ == "search":
        user_query_template = search_user_query_template
        query_keys_template = search_template
        cost = args.search_cost
        select_score = args.search_select_score
        max_action = 5
        if "[StopSearch]" in response:
            has_stop = True
    else:
        user_query_template = expand_user_query_template
        query_keys_template = expand_template
        cost = args.expand_cost
        select_score = args.expand_select_score
        max_action = 5
        if "[StopExpand]" in response:
            has_stop = True

    # parse the model output
    user_query = re.findall(user_query_template, response, flags=re.DOTALL)
    if len(user_query) > 0:
        user_query = user_query[0].strip()
    else:
        user_query = ""
    query_keys = [
        q.strip() for q in re.findall(query_keys_template, response, flags=re.DOTALL)
    ]
    searched_paper_set = set()

    for idx in range(max(max_action, len(query_keys))):
        score = -cost
        if idx < max_action:
            searched_papers, value_model_prompts, select_prompts = [], [], []

            # do search or expand
            if idx < len(query_keys):
                if typ == "search":
                    searched_papers = []
                    search_ids = google_search_arxiv_id(query_keys[idx])
                    if search_ids is not None:
                        for search_id in search_ids:
                            if search_id in paper_id:
                                searched_paper = search_paper_by_title(
                                    paper_id[search_id], paper_db
                                )
                                if searched_paper:
                                    searched_papers.append(searched_paper)
                        searched_papers = searched_papers[:MAX_PAPERS]
                else:
                    searched_papers = get_expand_papers(
                        query_keys[idx], f_paper, paper_db
                    )

            if len(searched_papers) > 0:
                results = []
                for searched_paper in searched_papers:
                    if keep_letters(searched_paper["title"]) in answer:
                        results.append({"prob": 1})
                    else:
                        results.append({"prob": 0})

                if args.use_selector:
                    for searched_paper in searched_papers:
                        select_prompts.append(
                            select_prompt.format(
                                title=searched_paper["title"],
                                abstract=searched_paper["abstract"],
                                user_query=user_query,
                            )
                        )
                    selector_results = call_selector(select_prompts)
                    for i in range(len(results)):
                        if selector_results[i]["prob"] > results[i]["prob"]:
                            results[i]["prob"] = selector_results[i]["prob"]

                # gen value model prompt
                all_prompts = []
                # log(searched_papers, results)
                for searched_paper, result in zip(searched_papers, results):
                    if keep_letters(searched_paper["title"]) not in searched_paper_set:
                        searched_paper_set.add(keep_letters(searched_paper["title"]))
                    else:
                        continue
                    if result["prob"] > 0.5:
                        score += select_score
                    value_model_prompt, paper = gen_value_model_prompt(
                        searched_paper["title"], user_query, paper_db
                    )
                    if value_model_prompt is None:
                        continue
                    all_prompts.append([result["prob"], paper, value_model_prompt])
                all_prompts.sort(key=lambda x: x[0], reverse=True)
                all_prompts = all_prompts[: MAX_PAPERS + 2]
                all_prompts.sort(key=lambda x: len(x[2][0]["content"]))
                for i in all_prompts[:MAX_PAPERS]:
                    value_model_prompts.append(i[2])
                    with lock:
                        all_papers.append(
                            i[0], i[1], i[2][:1], answer
                        )  # result, paper, prompt, answer

            # get value model score
            if args.use_vm:
                with lock:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        try:
                            future = executor.submit(
                                call_vm,
                                value_model_prompts,
                                tokenizer,
                                query_responses.device,
                                value_model,
                            )
                            result = future.result()
                            score += args.gamma1 * result
                        except Exception as e:
                            print(f"call value function error: {e}")

        if idx < len(query_keys):
            scores.append(max(min(score, 5), -args.search_cost))

    # score should be apply to each '[' (58) token
    score_tensor = torch.zeros(query_responses.shape[1] - context_length)
    score_idx = len(socres) - 1
    for i in range(-1, -score_tensor.shape[0] - 1, -1):
        if has_stop and query_responses[num][i] == 60:  # ']'(60)
            has_stop = False  # only one stop token is rewarded
            score_tensor[i] = cost  # reward for stop token
        if query_responses[num][i] == 58:
            if score_idx < 0:
                break
            score_tensor[i] = scores[score_idx]
            score_idx -= 1
            if score_idx < 0:
                break
    with lock:
        all_scores[num] = score_tensor


def rollout(
    query_responses,
    tokenizer,
    context_length,
    value_model,
    args,
    paper_db,
    paper_id,
    answers,
    papers,
    typ="search",
    return_new_query=True,
):

    # decode to strs
    query_response_strs = tokenizer.batch_decode(
        query_responses, skip_special_tokens=True
    )
    # log("query_response_strs", query_response_strs)
    all_papers, all_scores = [], {}
    lock = threading.Lock()

    # papers response, search paper and generate
    threads = []
    for num, response in enumerate(query_response_strs):
        f_paper = None
        if typ == "expand":
            f_paper = papers[num]
        thread = threading.Thread(
            target=response_handler,
            args=(
                num,
                response,
                all_papers,
                all_scores,
                lock,
                query_responses,
                tokenizer,
                context_length,
                value_model,
                args,
                paper_db,
                paper_id,
                typ,
                f_paper,
                [keep_letters(answer) for answer in answers[num]],
            ),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    torch.distributed.barrier()
    all_scores_list = []
    for i in range(query_rersponses.shape[0]):
        if i in all_scores:
            all_scores_list.append(all_scores[i])
        else:
            print("Error, index not in value")
            all_scores_list.append(
                torch.zeros(query_responses.shape[1] - context_length)
            )
    all_scores = torch.stack(all_scores_list).to(query_responses.device)

    if return_new_query:
        # hard-coded to return 6 items
        all_papers.sort(key=lambda x: x[0], reverse=True)
        next_data = []
        if len(all_papers) == 0:
            return None, None, all_scores, []
        while len(all_papers) < 6:
            all_papers += all_papers
        next_data = all_papers[:6]
        value_model_prompts = [i[2] for i in next_data]

        # tokenize一下，处理0条的情况
        input_ids = None
        if len(next_data) > 0:
            input_ids = tokenizer.apply_chat_template(
                value_model_prompts,
                tokenize=True,
                padding=True,
                truncation=True,
                max_length=992,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor(input_ids, device=query_responses.device)

        return (
            input_ids,
            [i[1] for i in next_data],
            all_scores,
            [i[3] for i in next_data],
        )

    else:
        return None, None, all_scores, []
