import argparse
import random

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler

from data_utils import *
from train_utils import *
from eval_utils import *

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

def eval_model(model, tokenizer, dataset, device, eval_batch_size, **decode_params):
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

    val_loader = DataLoader(dataset, batch_size=eval_batch_size, collate_fn=val_data_collator)

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

    ########### Set Padding side to right - for loss calculation
    tokenizer.padding_side = "right"

    data_collator=lambda data: {
        'input_ids': torch.cat([tokenizer(f['target'], return_tensors="pt", truncation=True, max_length=args.max_seq_length, padding="max_length")["input_ids"] for f in data], dim=0),
        'attention_mask': torch.cat([tokenizer(f['target'], return_tensors="pt", truncation=True, max_length=args.max_seq_length, padding="max_length")["attention_mask"] for f in data],dim=0),
        'labels': mask_label(torch.cat([tokenizer(f['target'], return_tensors="pt", truncation=True, max_length=args.max_seq_length, padding="max_length")["input_ids"] for f in data], dim=0), tokenizer.pad_token_id),
    }
    val_loader = DataLoader(dataset, batch_size=eval_batch_size, collate_fn=data_collator)

    # Calculate Perp Separately with eval_dataloader
    for ii, batch in enumerate(tqdm(val_loader)):
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
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"]
    }

    ########### BLEU
    for i in range(1,5):
        score_dict[f"bleu-{i}"]=compute_bleu(pair_preds,pair_refs,max_order=i)

    ########### Perplexity
    loss_mean=torch.stack(losses).mean()
    score_dict["perplexity"]=torch.exp(loss_mean).detach().cpu().item()

    return score_dict

def eval(args):
    data_dir=DATA_DIRS[args.data_name]

    if args.input_mode=="":
        #Use Default Input Mode (All)
        input_mode=INPUT_MODE[args.data_name]
    else:
        input_mode=args.input_mode

    ########### Prepare Model
    if torch.cuda.is_available():
        num_gpus=torch.cuda.device_count()
    else:
        num_gpus=1
    
    # total_batch_size=args.gradient_accumulation_steps*args.batch_size*num_gpus
    total_batch_size=args.gradient_accumulation_steps*args.train_batch_size*num_gpus

    if args.only_weakener:
        data_option="_weakener"
    else:
        data_option=""
    
    total_batch_size=64
    run_name=f"{args.data_name}{data_option}_{input_mode}_{args.model_name}_batch{total_batch_size}_lr{args.lr}_seed{args.seed}"
    if args.fp16:
        run_name+="_fp16"
        
    ########### Prepare Data
    weights_dir=f'{args.output_dir}/{args.input_mode}/{run_name}'
    sub_dirs=[x[0] for x in os.walk(weights_dir)]
    print(sub_dirs)

    decode_params={
                    'num_beams': 5,
                    # 'top_k': 40,
                    # 'repetition_penalty': 1.2,                    
                    # 'no_repeat_ngram_size': 2,
                    # length_penalty=1.5, # 1
                    # 'early_stopping': True,
                    'max_length': 128 + 32 ,
                }

    step=args.eval_step

    ########### Prepare Result Log File
    if not os.path.exists("./nlg_results/"):
        os.makedirs("./nlg_results/")

    results_dir=f"./nlg_results/{run_name}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if step==-1:
        run_results_dir=f"{results_dir}/epoch_end"
        weight_dir=f"{weights_dir}/epoch_end/"
        
    else:
        run_results_dir=f"{results_dir}/checkpoint-{step}"
        weight_dir=f"{weights_dir}/checkpoint-{step}"

    ########### Prepare Model & Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(weight_dir, max_length = args.max_seq_length)
    model = GPT2LMHeadModel.from_pretrained(weight_dir)

    ########### Load Model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # actual_gpu_num=torch.cuda.device_count()
    # f = open (f'./parallelize_maps/gpt2_xl_{actual_gpu_num}gpu.json', "r")
    # map_config = json.loads(f.read())
    # device_map={}
    # for i in range(actual_gpu_num):
    #     device_map[i]=map_config[str(i)]
    # model.parallelize(device_map)

    model.to(device)

    
    ########### Prepare Data
    if args.data_name=="atomic":
        record_id_feature="AtomicEventId"
    else:
        record_id_feature="SNLIPairId"

    dev_dataset=DeltaGenTextDataset(file_path=f"{data_dir}/dev.jsonl", mode=input_mode, record_id_feature=record_id_feature, only_one_type=args.only_weakener)
    test_dataset=DeltaGenTextDataset(file_path=f"{data_dir}/test.jsonl", mode=input_mode, record_id_feature=record_id_feature, only_one_type=args.only_weakener)
    
    # print("Train Size",len(train_dataset))
    print("Dev Len",len(dev_dataset))
    print("Test Len",len(test_dataset))

    # Eval Dev
    dev_scores=eval_model(model, tokenizer, dev_dataset, device, args.eval_batch_size,**decode_params)
    
    # Eval Test
    test_scores=eval_model(model, tokenizer, test_dataset ,device, args.eval_batch_size,**decode_params)

    metrics=["perplexity","bleu-1","bleu-2","bleu-3","bleu-4","rouge-1","rouge-2","rouge-l"]

    with open(f"{run_results_dir}.csv","w") as f:
        f.write("data,perplexity,bleu-1,bleu-2,bleu-3,bleu-4,rouge-1,rouge-2,rouge-l\n")
        f.write("dev,"+",".join(["{:.4f}".format(dev_scores[m]) for m in metrics])+"\n")
        f.write("test,"+",".join(["{:.4f}".format(test_scores[m]) for m in metrics])+"\n")


def main(args):
    seed=args.seed
    set_seed(seed)

    #Start Train
    eval(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate finetuned GPT2 with strengthener/weakener')

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
    parser.add_argument('--only_weakener', action='store_true')


    # Hyperparameters
    parser.add_argument('--train_batch_size', type=int, help="Batch size", default=4)
    parser.add_argument('--eval_batch_size', type=int, help="Batch size", default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, help="gradient_accumulation_steps", default=1)
    parser.add_argument('--lr', type=float, help="Learning rate", default=1e-5)

    parser.add_argument('--eval_step', type=int, help="Checkpoint step to eval", default=0)

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
    
    parser.add_argument('--fp16', action='store_true')


    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
