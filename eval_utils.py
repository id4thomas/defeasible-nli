import datasets

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

#ROUGE Implementations
# https://github.com/pltrdy/rouge
from rouge import Rouge

# https://github.com/li-plus/rouge-metric
from rouge_metric import PyRouge

# load rouge for validation
# rouge = datasets.load_metric("rouge")


def compute_bleu(preds, refs, max_order=4, smooth=True):
    #Receive sentences -> Tokenize
    preds_tokenized=[pred.split() for pred in preds]
    
    #Check if multi-reference
    if isinstance(refs[0], str):
        #Single Reference
        # print("Single-Ref")
        refs_tokenized=[[ref.split()] for ref in refs]
    else:
        #Multi-reference
        # print("Multi-Ref")
        refs_tokenized=[[ref_sent.split() for ref_sent in ref] for ref in refs]
    # print(refs_tokenized)

    # bleu = datasets.load_metric("bleu")
    # scores=bleu.compute(predictions=preds_tokenized, references=refs_tokenized, max_order=max_order, smooth=smooth)

    order_weights=[
        (1,0,0,0),
        (0.5,0.5,0,0),
        (0.33,0.33,0.33,0),
        (0.25,0.25,0.25,0.25)
    ]
    scores=corpus_bleu(refs_tokenized, preds_tokenized, weights=order_weights[max_order-1])

    return scores

# multi_ref_mode: avg / best
# avg for summarization / best for answering
def compute_rouge(preds, refs, metrics=["rouge-l"], multi_ref_mode='best'):
    #Huggingface Datasets
    # rouge = datasets.load_metric("rouge")
    # scores=rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])

    #Available Metrics
    # rouge-1~5, rouge-l
    # rouge = Rouge(metrics=metrics)
    # scores=rouge.get_scores(preds, refs, avg=True)


    # mode: individual / average
    rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, mode='average',multi_ref_mode=multi_ref_mode )
    scores = rouge.evaluate(preds, refs)
    # print(scores)
    return scores