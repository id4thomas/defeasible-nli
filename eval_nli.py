import argparse
import random
import math

import numpy as np

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments

import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler
from torch.optim import Adam


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from data_utils import *

import wandb

DATA_DIRS={
    'snli': "../defeasible-nli/data/defeasible-nli/defeasible-snli",
    'atomic': "../defeasible-nli/data/defeasible-nli/defeasible-atomic",
    'social': "../defeasible-nli/data/defeasible-nli/defeasible-social"
}

INPUT_MODE={
    "snli":"phu",
    "atomic":"phu",
    "social":"hu"
}

# compute_metrics for Trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def num_correct(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def eval(model,dataloader,device):

    eval_correct = 0
    nb_eval_examples = 0

    eval_predictions = []
    eval_logits = []
    eval_pred_probs = []
    
    model.eval()
    for batch  in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label_ids = batch['labels'].to(device)

        with torch.no_grad():
            model_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
            tmp_eval_loss = model_output[0]
            logits = model_output[1]
        
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_correct = num_correct(logits, label_ids)

        eval_predictions.extend(np.argmax(logits, axis=1).tolist())
        eval_logits.extend(logits.tolist())
        # eval_pred_probs.extend([_compute_softmax(list(l)) for l in logits])

        eval_correct += tmp_eval_correct
        nb_eval_examples += input_ids.size(0)

    eval_accuracy = eval_correct / nb_eval_examples
    # print("Eval ACC:",eval_accuracy)
    # print(eval_correct,nb_eval_examples)
    return eval_logits,eval_accuracy, {'correct':eval_correct, 'total': nb_eval_examples}

def test(args):
     # model_name="roberta-base"
    model_name=args.model_name
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base",max_length = args.max_seq_length)
    model = RobertaForSequenceClassification.from_pretrained(args.model_dir)

    data_dir=DATA_DIRS[args.data_name]
    dev_dataset=DeltaATOMICDataset(tokenizer, file_path=f"{data_dir}/dev.jsonl", mode=INPUT_MODE[args.data_name])
    test_dataset=DeltaATOMICDataset(tokenizer, file_path=f"{data_dir}/test.jsonl", mode=INPUT_MODE[args.data_name])

    print("Dev Len",len(dev_dataset))
    print("Test Len",len(test_dataset))

    dev_dataloader=DataLoader(dev_dataset,sampler=SequentialSampler(dev_dataset),batch_size=args.batch_size)
    test_dataloader=DataLoader(test_dataset,sampler=SequentialSampler(test_dataset),batch_size=args.batch_size)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    _,dev_acc,dev_eval_details=eval(model,dev_dataloader,device)
    _,test_acc,test_eval_details=eval(model,test_dataloader,device)

    print("Dev Acc:",dev_acc)
    print(dev_eval_details)

    print("Test Acc:",test_acc)
    print(test_eval_details)


def main(args):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # wandb.init(project='defeasible_clf', entity='id4thomas')

    #Start Test
    test(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Finetune BERT model and save')

    # Path, Model Related
    parser.add_argument('--data_name',
                        type=str,
                        help="Dataset Name",
                        default="atomic")
    parser.add_argument('--data_dir', type=str, help='Location of data', default=None)
    parser.add_argument('--model_name',
                        type=str,
                        help="roberta model",
                        default="roberta-base")
    parser.add_argument('--model_dir',
                        type=str,
                        help="Model Checkpoint Dir",
                        default="./weights")

    # Other parameters
    parser.add_argument('--batch_size', type=int, help="Batch size", default=4)
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed")
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_samples', default=20, type=int)
    parser.add_argument('--no_tune_bert', action='store_true')

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
