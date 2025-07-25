# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import shutil
from typing import override, Optional

import torch
from accelerate import PartialState
from custom_agent.agent_dataset import AgentDataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    AutoConfig,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import torch.nn as nn
from peft import PeftModel
from transformers import Qwen2ForSequenceClassification, Qwen2Model


"""
python examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_tldr \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --eval_strategy steps \
    --eval_steps 100

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 100
"""


class FixZero3CheckpointPPOTrainer(PPOTrainer):

    @override
    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        backup_model = self.model
        self.model = self.model.policy

        Trainer.save_model(output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {
                name.removeprefix("policy."): param
                for name, param in state_dict.items()
                if name.startswith("policy.")
            }

        super()._save(output_dir, state_dict)


class CustomQwen2ForSequenceClassification(Qwen2ForSequenceClassification):
    def __init__(self, lora_path, config) -> None:
        super().__init__(config)
        self.num_labels = 1
        model = Qwen2Model(config)
        self.model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, 384, bias=False),
            nn.Linear(384, num_labels, bias=False),
        )

        self.post_init()


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    # torch_dtype = (
    #     model_args.torch_dtype
    #     if model_args.torch_dtype in ["auto", None]
    #     else getattr(torch, model_args.torch_dtype)
    # )
    # quantization_config = get_quantization_config(model_args)
    # model_kwargs = dict(
    #     revision=model_args.model_revision,
    #     attn_implementation=model_args.attn_implementation,
    #     torch_dtype=torch_dtype,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
        max_length=1024,
    )
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    # value_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path,
    #     trust_remote_code=model_args.trust_remote_code,
    #     num_labels=1,
    # )
    # for m in value_model.score.modules():
    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, mean=0, std=0.01)
    config = AutoConfig.from_pretrained(
        training_args.reward_model_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
    )
    value_model = CustomQwen2ForSequenceClassification(
        lora_path=training_args.lora_path, config=config
    )
    # 给score层初始化
    for m in value_model.score.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)

    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path,
    #     trust_remote_code=model_args.trust_remote_code,
    #     num_labels=1,
    # )
    policy_base = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    policy = PeftModel.from_pretrained(
        policy_base,
        adapter_name=training_args.model_adapter_name,
        torch_dtype=torch.bfloat16,
        model_id=training_args.lora_path,
        lora_dropout=0.05,
        is_trainable=True,
    )

    # 设置 ref_policy 为None， 只加载其LoRA Adapter
    ref_policy = None
    policy.load_adapter(
        training_args.lora_path,
        adapter_name=training_args.ref_adapter_name,
        is_trainable=False,
    )
    # ref_policy_base = AutoModelForCausalLM.from_pretrained(
    #     training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    # )
    # ref_policy = PeftModel.from_pretrained(
    #     ref_policy_base,
    #     adapter_name="ref_policy",
    #     torch_dtype=torch.bfloat16,
    #     model_id="/root/pasa/results/sft_crawler/checkpoint-3248",
    #     is_trainable=False,
    #     lora_dropout=0.0,
    # )

    # peft_config = get_peft_config(model_args)
    # if peft_config is None:
    #     ref_policy = AutoModelForCausalLM.from_pretrained(
    #         training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    #     )
    # else:  # peft
    #     ref_policy = None

    ################
    # Dataset
    ################
    train_dataset = AgentDataset(script_args.dataset_name, tokenizer)
    assert (
        train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id
    ), "The last token should not be an EOS token"
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # train_dataset = dataset[script_args.dataset_train_split]
    # eval_dataset = (
    #     dataset[script_args.dataset_test_split]
    #     if training_args.eval_strategy != "no"
    #     else None
    # )

    # def prepare_dataset(dataset, tokenizer):
    #     """pre-tokenize the dataset before training; only collate during training"""

    #     def tokenize(element):
    #         input_ids = tokenizer.apply_chat_template(
    #             element["messages"][:1],
    #             padding=False,
    #             add_generation_prompt=True,
    #         )
    #         return {"input_ids": input_ids, "lengths": len(input_ids)}

    #     return dataset.map(
    #         tokenize,
    #         remove_columns=dataset.column_names,
    #         num_proc=training_args.dataset_num_proc,
    #     )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    # with PartialState().local_main_process_first():
    #     train_dataset = prepare_dataset(train_dataset, tokenizer)
    #     if eval_dataset is not None:
    #         eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    #     # filtering
    #     train_dataset = train_dataset.filter(
    #         lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc
    #     )
    #     if eval_dataset is not None:
    #         eval_dataset = eval_dataset.filter(
    #             lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc
    #         )

    # assert (
    #     train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id
    # ), "The last token should not be an EOS token"
    ################
    # Training
    ################
    # trainer = PPOTrainer(
    #     args=training_args,
    #     processing_class=tokenizer,
    #     model=policy,
    #     ref_model=ref_policy,
    #     reward_model=reward_model,
    #     value_model=value_model,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     peft_config=peft_config,
    # )
    trainer = FixZero3CheckpointPPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        value_model=value_model,
        train_dataset=train_dataset,
        paper_db=training_args.paper_db,
        paper_id=training_args.paper_id,
        peft_config=None,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)

    # trainer.generate_completions()
