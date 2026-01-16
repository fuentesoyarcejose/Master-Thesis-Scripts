import os
import sys
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


LABEL2ID = {"Pro-Palestine": 0, "Neutral": 1, "Pro-Israel": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class TextDataset(torch.utils.data.Dataset):
    encodings: Dict[str, torch.Tensor]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"text": str, "label": int})
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df


def main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for stance classification (3-way)")
    parser.add_argument("--train_csv", default=os.path.join("Stance", "seed_labels.csv"), help="CSV with 'text' and 'label'")
    parser.add_argument("--model", default="roberta-base", help="HF model name")
    parser.add_argument("--output_dir", default=os.path.join("Stance", "stance_roberta"), help="Output dir for model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    df = load_data(args.train_csv)
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=args.val_size, random_state=args.seed, stratify=df["label"].tolist()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    val_enc = tokenizer(X_val, truncation=True, padding=True, max_length=512)

    train_ds = TextDataset({k: torch.tensor(v) for k, v in train_enc.items()}, y_train)
    val_ds = TextDataset({k: torch.tensor(v) for k, v in val_enc.items()}, y_val)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"f1_macro": macro_f1}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_res = trainer.evaluate()
    print(eval_res)

    # Detailed report
    preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    print(classification_report(y_val, preds, target_names=[ID2LABEL[0], ID2LABEL[1], ID2LABEL[2]]))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved best model to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


