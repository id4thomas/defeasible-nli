import json
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from file_utils import *

class DeltaATOMICDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path="train.jsonl", mode="phu",
                max_seq_len=256, is_predict=False):

        self.is_predict=is_predict

        assert os.path.isfile(file_path)

        records=read_jsonl_inputs(file_path)

        update_label_vals={
            "strengthener": 0,
            "weakener": 1,
        }

        self.inputs=[]
        self.labels=[]

        for record in tqdm(records):
            # Skip records with no update
            if record["UpdateTypeImpossible"]==True:
                continue
            premise=record["Premise"]
            hypothesis=record["Hypothesis"]
            update=record["Update"]

            # "concatenate sentences p,h,u separated by special token"
            update_tokens=tokenizer.tokenize(update)

            if mode=="phu":
                #premise;hypothesis;update
                tokenized_text=tokenizer(f"[premise] {premise} [hypo] {hypothesis} [update] {update}", padding="max_length")
            elif mode=="hu":
                #hypothesis;update
                tokenized_text=tokenizer(f"[hypo] {hypothesis} [update] {update}", padding="max_length")
            else:
                #update
                tokenized_text=tokenizer(f"[update] {update}", padding="max_length")

            self.inputs.append(tokenized_text)

            if not self.is_predict:
                update_type=record["UpdateType"]
                update_label=update_label_vals[update_type]
                self.labels.append(update_label)


    def __getitem__(self, idx):
        if self.is_predict:
            return {'input_ids': torch.tensor(self.inputs[idx]["input_ids"]),
                    'attention_mask': torch.tensor(self.inputs[idx]["attention_mask"])}
        else:
            return {'input_ids': torch.tensor(self.inputs[idx]["input_ids"]),
                    'attention_mask': torch.tensor(self.inputs[idx]["attention_mask"]),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

    def __len__(self):
        return len(self.inputs)