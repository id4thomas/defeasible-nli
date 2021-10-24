import argparse
import random
import math
import os

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

# import wandb
import matplotlib.pyplot as plt

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

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def eval_model(model,dataloader,device):

    eval_correct = 0
    nb_eval_examples = 0

    eval_labels=[]
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

        eval_labels.extend(label_ids.tolist())
        eval_predictions.extend(np.argmax(logits, axis=1).tolist())
        eval_logits.extend(logits.tolist())
        # eval_pred_probs.extend([_compute_softmax(list(l)) for l in logits])

        eval_correct += tmp_eval_correct
        nb_eval_examples += input_ids.size(0)

    eval_accuracy = eval_correct / nb_eval_examples
    # print("Eval ACC:",eval_accuracy)
    # print(eval_correct,nb_eval_examples)

    precision, recall, f1, _ = precision_recall_fscore_support(eval_labels, eval_predictions, average='binary')
    acc = accuracy_score(eval_labels, eval_predictions)

    # return eval_logits,eval_accuracy, {'correct':eval_correct, 'total': nb_eval_examples}
    return {'acc': acc,'precision':precision, 'recall': recall, 'f1': f1, 'correct':eval_correct, 'total': nb_eval_examples}

def eval(args,model_dir):
     # model_name="roberta-base"
    model_name=args.model_name
    tokenizer = RobertaTokenizer.from_pretrained(model_name,max_length = args.max_seq_length)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)

    data_dir=DATA_DIRS[args.data_name]
    dev_dataset=DeltaInferenceDataset(tokenizer, file_path=f"{data_dir}/dev.jsonl", mode=INPUT_MODE[args.data_name])
    test_dataset=DeltaInferenceDataset(tokenizer, file_path=f"{data_dir}/test.jsonl", mode=INPUT_MODE[args.data_name])

    print("Dev Len",len(dev_dataset))
    print("Test Len",len(test_dataset))

    dev_dataloader=DataLoader(dev_dataset,sampler=SequentialSampler(dev_dataset),batch_size=args.batch_size)
    test_dataloader=DataLoader(test_dataset,sampler=SequentialSampler(test_dataset),batch_size=args.batch_size)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    dev_results=eval_model(model,dev_dataloader,device)
    test_results=eval_model(model,test_dataloader,device)

    print("Dev Acc:",dev_results["acc"])
    print(dev_results)

    print("Test Acc:",test_results["acc"])
    print(test_results)
    
    return dev_results,test_results

def main(args):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # wandb.init(project='defeasible_clf', entity='id4thomas')

    model_name=args.model_name
    total_batch_size=args.gradient_accumulation_steps*args.batch_size
    run_name=f"{args.data_name}_{args.model_name}_batch{total_batch_size}_lr{args.lr}_seed{args.seed}"
    
    weight_dir=f'{args.output_dir}/{run_name}/'
    sub_dirs=[x[0] for x in os.walk(weight_dir)]
    print(sub_dirs)
    
    save_steps=[]
    for sub_dir in sub_dirs:
        step=sub_dir.split("/")[-1].split('-')[-1]
        if step=="":
            continue
        save_steps.append(int(step))

    dev_results_all=[]
    test_results_all=[]
    for i,step in  enumerate(save_steps):
        print("Epoch",i+1,step)
        #Start Test
        dev_results,test_results=eval(args,f"{weight_dir}/checkpoint-{step}")
        dev_results_all.append(dev_results)
        test_results_all.append(test_results)

    if not os.path.exists(f"./nli_results/plot/{run_name}/"):
        os.makedirs(f"./nli_results/plot/{run_name}/")

    # if not os.path.exists("./nli_results/perf/{run_name}/"):
    #     os.makedirs("./nli_results/perf/{run_name}/")

    epochs=range(1,len(save_steps)+1)
    perf_metrics=["acc","precision","recall","f1"]

    for perf_metric in perf_metrics:
        #Plot Acc
        plt.plot(epochs,[d[perf_metric] for d in dev_results_all],color="blue",label="dev")
        plt.plot(epochs,[d[perf_metric] for d in test_results_all],color="red",label="test")
        plt.legend()
        plt.savefig(f"./nli_results/plot/{run_name}/{perf_metric}.png")
        plt.clf()

    with open(f"./nli_results/{run_name}.csv",'w') as f:
        f.write("epoch,dev_acc,dev_precision,dev_recall,dev_f1,test_acc,test_precision,test_recall,test_f1\n")
        for epoch,dev,test in zip(epochs,dev_results_all,test_results_all):
            f.write(f"{epoch},")
            # for perf_metric in perf_metrics:
            # f.write("{},{},{},{},".format(dev["acc"],dev["p"],dev["r"],dev["f"]))
            f.write("{},{},{},{},".format(dev["acc"],dev["precision"],dev["recall"],dev["f1"]))
            f.write("{},{},{},{}\n".format(test["acc"],test["precision"],test["recall"],test["f1"]))

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
    parser.add_argument('--output_dir',
                        type=str,
                        help="Model Checkpoint Dir",
                        default="./weights")


    # parser.add_argument('--metrics_out_file', default="metrics.json")

    # Hyperparameters
    parser.add_argument('--epochs', type=int, help="Num epochs", default=3)
    parser.add_argument('--batch_size', type=int, help="Batch size", default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, help="gradient_accumulation_steps", default=1)
    parser.add_argument('--lr', type=float, help="Learning rate", default=1e-5)
    parser.add_argument('--training_data_fraction', type=float, default=1.0)

    parser.add_argument('--warmup_proportion',
                        type=float,
                        default=0.2,
                        help="Portion of training to perform warmup")


    # Other parameters
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
