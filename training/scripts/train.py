import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "training" / "datasets"


@dataclass
class Config:
    model_name: str
    train_file: str
    validation_file: str
    output_dir: str

    lora_r: int
    lora_alpha: int
    lora_dropout: float

    epochs: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int

    learning_rate: float
    warmup_ratio: float
    weight_decay: float

    max_length: int

    min_pixels: int
    max_pixels: int

    attn_implementation: str = "sdpa"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class FGS3VLDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        processor,
        max_length: int,
    ):
        annotation_path = Path(annotation_file)

        if not annotation_path.is_absolute():
            annotation_path = ROOT / annotation_path

        if not annotation_path.is_file():
            raise FileNotFoundError(f"데이터셋 파일이 없습니다: {annotation_path}")

        self.items = load_jsonl(annotation_path)
        self.dataset_dir = annotation_path.parent

        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def resolve_image_path(self, item) -> Path:
        image_value = item.get("image")

        if not image_value:
            raise ValueError(f"image 경로가 없습니다: {item.get('id', '?')}")

        image_path = Path(image_value)

        if image_path.is_absolute():
            candidates = [image_path]
        else:
            candidates = [
                self.dataset_dir / image_path,
                DATASET_DIR / image_path,
                ROOT / image_path,
            ]

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다: {image_value} (기준: {self.dataset_dir})"
        )

    def message_content(self, item, role: str):
        for message in item.get("messages", []):
            if message.get("role") != role:
                continue

            content = message.get("content", "")

            if isinstance(content, list):
                return content

            return [{"type": "text", "text": str(content)}]

        raise ValueError(f"'{role}' 메시지가 없습니다: {item.get('id', '?')}")

    def build_messages(self, item, image: Image.Image):
        return [
            {
                "role": "system",
                "content": self.message_content(item, "system"),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    *self.message_content(item, "user"),
                ],
            },
            {
                "role": "assistant",
                "content": self.message_content(item, "assistant"),
            },
        ]

    def build_prompt_messages(self, item, image: Image.Image):
        return self.build_messages(item, image)[:-1]

    def __getitem__(self, index):
        item = self.items[index]

        with Image.open(self.resolve_image_path(item)) as opened:
            image = opened.convert("RGB")

        full_messages = self.build_messages(item, image)
        prompt_messages = self.build_prompt_messages(item, image)

        full = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
        )

        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]

        prompt_length = prompt["input_ids"].shape[-1]

        labels = input_ids.clone()

        labels[:prompt_length] = -100

        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        if input_ids.shape[-1] > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for key in [
            "pixel_values",
            "image_grid_thw",
        ]:
            if key in full:
                result[key] = full[key]

        return result


@dataclass
class DataCollator:
    processor: Any

    def __call__(self, features):
        batch = {}

        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )

        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0,
        )

        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=-100,
        )

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"] for f in features],
                dim=0,
            )

        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"] for f in features],
                dim=0,
            )

        return batch


def apply_lora(model, config: Config):
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
    )

    parser.add_argument(
        "--train",
        default="training/datasets/train.jsonl",
    )

    parser.add_argument(
        "--validation",
        default="training/datasets/validation.jsonl",
    )

    parser.add_argument(
        "--output",
        default="training/outputs/fgs-qwen3-vl-lora",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
    )

    # sdpa: PyTorch 내장, 설치 필요 없음 (DGX Spark는 flash-attn 설치 파일이 없음)
    # flash_attention_2: flash-attn 패키지가 설치돼 있을 때만
    parser.add_argument(
        "--attn",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        model_name=args.model,
        train_file=args.train,
        validation_file=args.validation,
        output_dir=args.output,

        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,

        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,

        max_length=args.max_length,

        min_pixels=401408,
        max_pixels=1003520,

        attn_implementation=args.attn,
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )

    print("Loading model...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
    )

    model.config.use_cache = False

    model.enable_input_require_grads()

    model.gradient_checkpointing_enable()

    model = apply_lora(
        model,
        config,
    )

    train_dataset = FGS3VLDataset(
        config.train_file,
        processor,
        config.max_length,
    )

    validation_dataset = FGS3VLDataset(
        config.validation_file,
        processor,
        config.max_length,
    )

    collator = DataCollator(
        processor=processor,
    )

    # 최신 transformers는 warmup_ratio를 받지 않으므로 비율을 step 수로 직접 바꾼다.
    # (모든 버전에서 warmup_steps는 동작함)
    steps_per_epoch = math.ceil(
        len(train_dataset)
        / (config.batch_size * config.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = math.ceil(total_steps * config.warmup_ratio)
    print(f"총 {total_steps} step, warmup {warmup_steps} step")

    training_args = TrainingArguments(
        output_dir=config.output_dir,

        num_train_epochs=config.epochs,

        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,

        gradient_accumulation_steps=(
            config.gradient_accumulation_steps
        ),

        learning_rate=config.learning_rate,

        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,

        lr_scheduler_type="cosine",

        bf16=True,

        gradient_checkpointing=True,

        logging_steps=1,

        save_steps=100,
        eval_steps=100,

        eval_strategy="steps",
        save_strategy="steps",

        save_total_limit=3,

        remove_unused_columns=False,

        report_to="none",

        optim="adamw_torch",

        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        data_collator=collator,

    )

    print()
    print("Starting training...")
    print()

    trainer.train()

    print()
    print("Saving LoRA adapter...")

    trainer.save_model(
        config.output_dir
    )

    processor.save_pretrained(
        config.output_dir
    )

    print()
    print(
        f"Saved to: {config.output_dir}"
    )


if __name__ == "__main__":
    main()