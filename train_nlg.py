import argparse

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from transformers import Trainer, TrainingArguments, TrainerCallback

from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler


from data_utils2 import *
from train_utils import *
from eval_utils import *

import wandb

from nltk.tokenize import sent_tokenize

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

########### Mask Label for loss calculation
def mask_label(labels, pad_token_id):
    masked_labels = labels.clone()
    masked_labels[labels==pad_token_id] = -100
    return masked_labels

class GenEvalCallback(TrainerCallback):
    # Custom Evlaute Callback
    def on_evaluate(self, args, state, control, model, tokenizer, eval_dataloader, **kwargs):
        # print("ON EVALUATE")
        model.eval()

        ########### Set Padding side to left - for batch generation
        tokenizer.padding_side = "left" 

        val_data_collator=lambda data: {
            'input_ids': torch.cat([tokenizer(f['query'], return_tensors="pt", truncation=True, max_length=128, padding="max_length")["input_ids"] for f in data], dim=0),
            'attention_mask': torch.cat([tokenizer(f['query'], return_tensors="pt", truncation=True, max_length=128, padding="max_length")["attention_mask"] for f in data], dim=0),
            'updates': [f["update"] for f in data],
            'targets': [f["target"] for f in data],
            'record_id': [f["record_id"] for f in data],
            'update_type': [f["update_type"] for f in data],
        }

        val_loader = DataLoader(eval_dataloader.dataset, batch_size=16, collate_fn=val_data_collator)

        decode_params={
                    'num_beams': 5,
                    # 'top_k': 40,
                    # 'repetition_penalty': 1.2,                    
                    # 'no_repeat_ngram_size': 2,
                    # length_penalty=1.5, # 1
                    # 'early_stopping': True,
                    'max_length': 128 + 32 ,
                }

        preds=[]
        answers=[]
        record_ids=[]

        losses=[]

        device = 'cuda:0'
        for ii, batch in enumerate(tqdm(val_loader)):
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            weakeners=batch["updates"]
            record_id=batch["record_id"]

            # print("INPUT IDS",input_ids)

            answers.extend(weakeners)
            record_ids.extend(record_id)

            with torch.no_grad():
                outs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_generate=1,
                    pad_token_id=tokenizer.pad_token_id,
                    **decode_params
                )

                gens=[]
                for gen in outs:
                    # print("GEN TOKEN",gen)
                    gens.append(tokenizer.decode(gen,skip_special_tokens=True))

                #Get Generated
                for gen in gens:
                    generated=gen.split("[weakener]")[-1][1:]
                    # print("GEN:",gen)
                    # print("=================================")
                    if len(generated)==0:
                        generated="None"
                    else:
                        sents=sent_tokenize(generated)
                        if len(sents)==0:
                            generated="None"
                        else:
                            generated=sents[0]
                        
                    # print("FINAL:",generated)
                    preds.append(generated)

        # Turn Tokenizer padding back
        tokenizer.padding_side = "right"

        # Calculate Perp Separately with eval_dataloader
        for ii, batch in enumerate(tqdm(eval_dataloader)):
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            labels=batch["labels"].to(device)
            # print(labels)
             #Batch Calc
            with torch.no_grad():
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(output.loss)

        print("PREDS",preds[:20])
        print("ANSWERS",answers[:20])

        ########### Make (Prediction,References) pairs (Multi-reference)
        ref_pred_pairs={}
        for gen,ans,pair_id in zip(preds,answers,record_ids):
            if pair_id not in ref_pred_pairs.keys():
                ref_pred_pairs[pair_id]=[[],[]]
                ref_pred_pairs[pair_id][1].append(gen)

            #Refs
            ref_pred_pairs[pair_id][0].append(ans)

        pair_refs=[]
        pair_preds=[]
        for key in ref_pred_pairs.keys():
            pair_refs.append([ref for ref in ref_pred_pairs[key][0]])
            pair_preds.append(ref_pred_pairs[key][1][0])

        ########### ROUGE (multi-reference)
        rouge_scores=compute_rouge(pair_preds,pair_refs,metrics=["rouge-1","rouge-2","rouge-l"])

        score_dict = {
            "val rouge-1": rouge_scores["rouge-1"]["f"],
            "val rouge-2": rouge_scores["rouge-2"]["f"],
            "val rouge-l": rouge_scores["rouge-l"]["f"]
        }

        ########### BLEU
        for i in range(1,5):
            score_dict[f"val bleu-{i}"]=compute_bleu(pair_preds,pair_refs,max_order=i)

        ########### Perplexity
        loss_mean=torch.stack(losses).mean()
        score_dict["val perplexity"]=torch.exp(loss_mean).detach().cpu().item()
        
        ########### Log to Wandb
        print(score_dict)
        wandb.log(score_dict)

        ########### Save Model Weights
        steps=state.global_step
        save_dir=f"{args.output_dir}/checkpoint-{steps}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

def load_model_from_checkpoint(checkpoint_dir,epoch):
    #Model
    model = GPT2LMHeadModel.from_pretrained(f"{checkpoint_dir}/epoch_{epoch}")

    #Tokenizer
    tokenizer=GPT2Tokenizer.from_pretrained(f"{checkpoint_dir}/epoch_{epoch}")
    return model,tokenizer

def train(args):
    ########### Get Model Params
    model_name=args.model_name
    data_dir=DATA_DIRS[args.data_name]

    if args.input_mode=="":
        #Use Default Input Mode (All)
        input_mode=INPUT_MODE[args.data_name]
    else:
        input_mode=args.input_mode

    if torch.cuda.is_available():
        num_gpus=torch.cuda.device_count()
        if args.parallelize:
            num_gpus=1
    else:
        num_gpus=1
    
    if args.only_weakener:
        run_type="_weakener"
    else:
        run_type=""

    ########### Prepare Data
    if args.data_name=="atomic":
        record_id_feature="AtomicEventId"
    else:
        record_id_feature="SNLIPairId"
        
    train_dataset=DeltaGenTextDataset(file_path=f"{data_dir}/train.jsonl", mode=input_mode, record_id_feature=record_id_feature, only_one_type=args.only_weakener)
    dev_dataset=DeltaGenTextDataset(file_path=f"{data_dir}/dev.jsonl", mode=input_mode, record_id_feature=record_id_feature, only_one_type=args.only_weakener)
    
    print("Train Size",len(train_dataset))
    print("Dev Len",len(dev_dataset))
    
    total_batch_size=args.gradient_accumulation_steps*args.batch_size*num_gpus
    run_name=f"{args.data_name}{run_type}_{input_mode}_{args.model_name}_batch{total_batch_size}_lr{args.lr}_seed{args.seed}"
    if args.fp16:
        run_name+="_fp16"

    wandb.run.name=run_name

    ########### Prepare Model & Tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token='<|pad|>')
    tokenizer.add_tokens(["[premise]","[hypo]","[strengthener]","[weakener]"])
    model.resize_token_embeddings(len(tokenizer))

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    train_data_collator=lambda data: {
        'input_ids': torch.cat([tokenizer(f['target'], return_tensors="pt", truncation=True, max_length=args.max_seq_length, padding="max_length")["input_ids"] for f in data], dim=0),
        'attention_mask': torch.cat([tokenizer(f['target'], return_tensors="pt", truncation=True, max_length=args.max_seq_length, padding="max_length")["attention_mask"] for f in data],dim=0),
        'labels': mask_label(torch.cat([tokenizer(f['target'], return_tensors="pt", truncation=True, max_length=args.max_seq_length, padding="max_length")["input_ids"] for f in data], dim=0), pad_token_id),
    }

    ########### Load Model to GPU
    if args.parallelize:
        actual_gpu_num=torch.cuda.device_count()
        f = open (f'./parallelize_maps/gpt2_xl_{actual_gpu_num}gpu.json', "r")
        map_config = json.loads(f.read())
        device_map={}
        for i in range(actual_gpu_num):
            device_map[i]=map_config[str(i)]
        model.parallelize(device_map)


    if not os.path.exists(f'{args.output_dir}/{input_mode}/{run_name}'):
        os.makedirs(f'{args.output_dir}/{input_mode}/{run_name}')

    training_args = TrainingArguments(
        output_dir = f'{args.output_dir}/{input_mode}/{run_name}',
        run_name = run_name,
        num_train_epochs=args.epochs,
        # Batch Size
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size= args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,    
        learning_rate=args.lr,
        # warmup_steps=500,
        # weight_decay=0.01,
        #SAVE
        save_strategy="epoch",
        #EVAL
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        #LOG
        logging_dir = f"./log_nlg/{run_name}",
        logging_steps = 10,
        disable_tqdm = False, 
        seed=args.seed,
        report_to="wandb",
        fp16 = args.fp16,
        # dataloader_num_workers = 8,
    )

    #Log Memory Usage before Train
    wandb.log({"GPU Memory":torch.cuda.memory_allocated()})
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=train_data_collator
    )
    print("IS PARALLEL",trainer.is_model_parallel)

    # Add Evaluation Callback
    trainer.add_callback(GenEvalCallback)

    if args.is_resume:
        trainer.train(resume_from_checkpoint=f'{args.output_dir}/{input_mode}/{run_name}/epoch_{args.resume_epoch}')
    else:
        trainer.train()

    #SAVE TRAINER STATE
    trainer.save_state()
        

def main(args):
    seed=args.seed
    set_seed(seed)

    #Get Wandb ID
    wandb_id = wandb.util.generate_id()
    print("Wandb ID",wandb_id)

    if args.is_resume:
        wandb.init(project='defeasible_gen2', entity='id4thomas',resume=True)
    else:
        wandb.init(id=wandb_id, project='defeasible_gen2', entity='id4thomas')

    #Start Train
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Finetune GPT2 to generate strengthener/weakener')

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
    parser.add_argument('--input_mode',type=str,default="")
    parser.add_argument('--only_weakener', action='store_true', help="Train with only weakener samples")


    # parser.add_argument('--metrics_out_file', default="metrics.json")

    # Hyperparameters
    parser.add_argument('--epochs', type=int, help="Num Epohcs", default=1)
    parser.add_argument('--batch_size', type=int, help="Batch size", default=4)
    parser.add_argument('--lr', type=float, help="Learning rate", default=1e-5)
    parser.add_argument('--training_data_fraction', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, help="gradient_accumulation_steps", default=1)

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

    parser.add_argument("--eval_steps", type=int, default=40)

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--parallelize', action='store_true',help="Use Huggingface parallelize")

    #Resuming Training
    parser.add_argument('--is_resume', action='store_true')
    parser.add_argument('--resume_epoch', type=int, help="Epoch to resume from", default=1)


    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
