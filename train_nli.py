import argparse
import random

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments

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

def train(args):
    # model_name="roberta-base"
    model_name=args.model_name
    tokenizer = RobertaTokenizer.from_pretrained(model_name,max_length = args.max_seq_length)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    data_dir=DATA_DIRS[args.data_name]
    train_dataset=DeltaInferenceDataset(tokenizer, file_path=f"{data_dir}/train.jsonl", mode=INPUT_MODE[args.data_name])
    dev_dataset=DeltaInferenceDataset(tokenizer, file_path=f"{data_dir}/dev.jsonl", mode=INPUT_MODE[args.data_name])

    print("Train Size",len(train_dataset))
    print("Dev Len",len(dev_dataset))

    num_gpus=2
    total_batch_size=args.gradient_accumulation_steps*args.batch_size*num_gpus

    run_name=f"{args.data_name}_{args.model_name}_batch{total_batch_size}_lr{args.lr}_seed{args.seed}"
    batch_size=args.batch_size
    epochs=args.epochs

    training_args = TrainingArguments(
        output_dir = f'{args.output_dir}/{run_name}',
        run_name = run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size= batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps = args.gradient_accumulation_steps,    
        # load_best_model_at_end=True,
        # warmup_steps=500,
        # weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy = "epoch",
        # evaluation_strategy="steps",
        # eval_steps=100,
        logging_dir = f"./log/{run_name}",
        logging_steps = 8,
        disable_tqdm = False, 
        seed=args.seed,
        report_to="wandb"
        # fp16 = args.fp16,
        # dataloader_num_workers = 8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    #Save Model
    # trainer.save_model(save_dir)


def main(args):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # wandb.init(project='defeasible_clf', entity='id4thomas')

    #Start Train
    train(args)


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
