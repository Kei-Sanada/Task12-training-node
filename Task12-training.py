import os
from dataclasses import dataclass
import time  # 時間計測用

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template

# Using RTX 6000 Ada 48GB

@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int


class TimeoutCallback:
    """2時間制限のためのコールバック"""
    def __init__(self, max_time_hours=2):
        self.max_time_seconds = max_time_hours * 3600
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_time_seconds:
            print(f"Time limit reached: {elapsed_time/3600:.2f} hours")
            control.should_training_stop = True


def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments
):
    print(f"Starting training with {model_id}")
    print(f"Training configuration: epochs={training_args.num_train_epochs}, batch_size={training_args.per_device_train_batch_size}")
    
    start_time = time.time()
    assert model_id in model2template, f"model_id {model_id} not supported"
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=5e-5,  # Qwen2.5に最適化
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="adamw_torch_fused",  # 高速化のためfused optimizerを使用
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
        dataloader_num_workers=4,  # データローディング高速化
        dataloader_pin_memory=True,  # メモリ効率化
        gradient_checkpointing=True,  # メモリ節約
        save_steps=500,  # 保存頻度を調整
        eval_steps=500,  # 評価頻度を調整
        warmup_ratio=0.1,  # warmupの比率
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Load dataset
    dataset = SFTDataset(
    file="https://fed-ledger-prod-dataset.s3.amazonaws.com/12/training_set.jsonl?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASSFQ745NLT5K57N2%2F20250716%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20250716T144600Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=afe57cd54ce40999e4e901ab8ced192a471cc0c4309dfa8da670ee77d9f69ebd",
    tokenizer=tokenizer,
    max_seq_length=context_length,
    template=model2template[model_id],
    )

    # Define trainer
    timeout_callback = TimeoutCallback(max_time_hours=2)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
        callbacks=[timeout_callback],  # 時間制限コールバックを追加
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder (Windows/Linux互換)
    import platform
    if platform.system() == "Windows":
        os.system("rmdir /s /q outputs\\checkpoint-* 2>nul")
    else:
        os.system("rm -rf outputs/checkpoint-*")

    # 学習時間を記録
    end_time = time.time()
    training_time = (end_time - start_time) / 3600
    print(f"Training Completed in {training_time:.2f} hours")


if __name__ == "__main__":
    # Define training arguments for LoRA fine-tuning
    training_args = LoraTrainingArguments(
        num_train_epochs=5,  # 2時間制限内で効果的な学習
        per_device_train_batch_size=4,  # Qwen2.5 7Bに最適化
        gradient_accumulation_steps=2,
        lora_rank=32,  # より表現力のあるLoRA
        lora_alpha=64,  # rankの2倍に設定
        lora_dropout=0.1,
    )

    # Set model ID and context length
    model_id = "Qwen/Qwen2.5-7B-Instruct"  # Qwen2.5 7Bに変更
    context_length = 2048

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id, context_length=context_length, training_args=training_args
    )
