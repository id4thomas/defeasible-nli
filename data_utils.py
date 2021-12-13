import json
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from file_utils import *

def records_to_dict(file_path,mode="phu"):
    assert os.path.isfile(file_path)

    records=read_jsonl_inputs(file_path)

    update_label_vals={
        "strengthener": 0,
        "weakener": 1,
    }

    record_dict={
        'premise':[],
        'hypothesis':[],
        'update':[],
        'update_type':[]
    }

    for record in tqdm(records):
        # Skip records with no update
        if record["UpdateTypeImpossible"]==True:
            continue
        hypothesis=record["Hypothesis"]
        update=record["Update"]
        update_type=record["UpdateType"]

        if mode=="phu":
            premise=record["Premise"]
            #premise;hypothesis;update   
            record_dict["premise"].append(premise)

        record_dict["hypothesis"].append(hypothesis)
        record_dict["update"].append(update)
        record_dict["update_type"].append(update_type)

    return record_dict

class DeltaInferenceDataset(torch.utils.data.Dataset):
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


class DeltaGenTextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path="train.jsonl", mode="phu", only_one_type=False, train_update_type="weakener", record_id_feature="AtomicEventId"):
        
        assert os.path.isfile(file_path)

        records=read_jsonl_inputs(file_path)#[:50]

        #Sample
        # if "dev" in file_path:
            # records=records[:int(len(records)*0.03)]
        # records=records[:int(len(records)*0.03)]

        self.inputs=[]
        self.labels=[]

        for record in tqdm(records):
            # Skip records with no update
            if record["UpdateTypeImpossible"]==True:
                continue
            
            if only_one_type:
                if record["UpdateType"]==train_update_type:
                    pass
                else:
                    continue
                
            hypothesis=record["Hypothesis"]
            update=record["Update"]
            update_type=record["UpdateType"]

            if mode=="phu":
                premise=record["Premise"]
                #premise;hypothesis;update
                query=f"<|startoftext|>[premise] {premise} [hypo] {hypothesis} [{update_type}]"
                target=f"<|startoftext|>[premise] {premise} [hypo] {hypothesis} [{update_type}] {update}<|endoftext|>"
            elif mode=="hu":
                #hypothesis;update
                query=f"<|startoftext|>[hypo] {hypothesis} [{update_type}]"
                target=f"<|startoftext|>[hypo] {hypothesis} [{update_type}] {update}<|endoftext|>"
            else:
                #update
                query=f"<|startoftext|>[{update_type}]"
                target=f"<|startoftext|>[{update_type}] {update}<|endoftext|>"

            self.inputs.append([
                query,
                target,
                update,
                record[record_id_feature],
                update_type
            ])


    def __getitem__(self, idx):
        return {'query': self.inputs[idx][0],
                'target': self.inputs[idx][1],
                'update': self.inputs[idx][2],
                'record_id': self.inputs[idx][3],
                'update_type': self.inputs[idx][4],
        }

    def __len__(self):
        return len(self.inputs)

# class DeltaGenEncoderDecoderDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer, file_path="train.jsonl", mode="phu",
#                 max_seq_len=128, is_predict=False):

#         self.is_predict=is_predict

#         assert os.path.isfile(file_path)

#         records=read_jsonl_inputs(file_path)

#         update_label_vals={
#             "strengthener": 0,
#             "weakener": 1,
#         }

#         self.inputs=[]
#         self.labels=[]

#         for record in tqdm(records):
#             # Skip records with no update
#             if record["UpdateTypeImpossible"]==True:
#                 continue

#             hypothesis=record["Hypothesis"]
#             update=record["Update"]
#             update_type=record["UpdateType"]

#             # "concatenate sentences p,h,u separated by special token"
#             update_tokens=tokenizer.tokenize(update)

#             if mode=="phu":
#                 premise=record["Premise"]
#                 #premise;hypothesis;update
#                 enc_tokenized_text=tokenizer(f"[premise] {premise} [hypo] {hypothesis} [{update_type}]", truncation=True,max_length=max_seq_len, padding="max_length")
#                 dec_tokenized_text=tokenizer(f"{update}", truncation=True,max_length=max_seq_len, padding="max_length")
#             elif mode=="hu":
#                 #hypothesis;update
#                 tokenized_text=tokenizer(f"[hypo] {hypothesis} [{update_type}] {update}", padding="max_length")
#             else:
#                 #update
#                 tokenized_text=tokenizer(f"[{update_type}] {update}", padding="max_length")

#             self.inputs.append((enc_tokenized_text,dec_tokenized_text))



#     def __getitem__(self, idx):
#         return {'input_ids': torch.tensor(self.inputs[idx][0]["input_ids"]),
#                 'labels': torch.tensor(self.inputs[idx][1]["input_ids"]),
#                 'attention_mask': torch.tensor(self.inputs[idx][0]["attention_mask"])}

#     def __len__(self):
#         return len(self.inputs)