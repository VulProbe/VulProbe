import argparse
import logging
import sys
import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
from unixcoder_model import UnixCoderForVulnerabilityDetection
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        if 'csv' in file_path:
            df = pd.read_csv(file_path)
        if 'jsonl' in file_path:
            df = pd.read_json(file_path, orient="records", lines=True)
        funcs = df["src"].tolist()
        labels = df["label"].tolist()
        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def convert_examples_to_features(func, label, tokenizer, args):
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[1,3])

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0

    model.zero_grad()

    for idx in range(args.epochs):
        logger.info("***** Epoch %d *****", idx)
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (input_ids, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits, _, _ = model(input_ids=input_ids, labels=labels, output_attentions=True)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model, device_ids=[1,3])

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        (input_ids, labels)=[x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit, _, _ = model(input_ids=input_ids, labels=labels, output_attentions=True)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #calculate scores
    logits = np.concatenate(logits,0)
    y_trues = np.concatenate(y_trues,0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:,1]>best_threshold

    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[1,3])

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in test_dataloader:
        (input_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit, _, _ = model(input_ids=input_ids, labels=labels, output_attentions=True)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:,1]>best_threshold
        
    
    test_result_df = pd.DataFrame({
        "label": y_trues.tolist(),
        "pred": y_preds.tolist()
    })
    # test_result_df.to_csv(f'test_result_{args.seed}.csv', index=False)
    
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold":best_threshold,
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    logits = [l for l in logits]
    result_df = generate_result_df(logits, y_trues, y_preds, args)
    
    # Do explanation and save the scores
    if args.do_explaination:
        methods = []
        if args.explaination_method == 'all':
            methods = ['attention', 'deeplift_shap', 'gradient_shap', 'saliency']
        else:
            methods = [args.explaination_method]
        for reasoning_method in methods:
            logger.info(f"***** Explaining the Model with {reasoning_method} *****")
            
            
            correct_indices = np.where((y_trues == y_preds))
            correct_indices = list(correct_indices[0])
            print("correct prediction count: ", len(correct_indices))
            
            tp_indices = np.where((y_trues == y_preds) & (y_trues == 1))
            tp_indices = list(tp_indices[0])
            print("correct prediction vulnerable count: ", len(tp_indices))
            
            # batch_size = 1
            dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, num_workers=1)
            progress_bar = tqdm(dataloader, total=len(dataloader))
            # index of result_df
            index = 0
            for batch in progress_bar:
                (input_ids, labels) = batch
                ids = input_ids[0].detach().tolist()
                all_tokens = tokenizer.convert_ids_to_tokens(ids)
                all_tokens = [token.replace("Ġ", "") for token in all_tokens]
                all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
                
                flaw_lines = result_df.iloc[index]["flaw_line"]
                flaw_line_seperator = "/~/"
                flaw_lines = get_all_flaw_lines(flaw_lines=flaw_lines, flaw_line_seperator=flaw_line_seperator)
                flaw_tokens_encoded = encode_all_lines(all_lines=flaw_lines, tokenizer=tokenizer)
                verified_flaw_lines = []
                do_explaination = False
                
                # if len(flaw_tokens_encoded) > 0:
                #     print()
                #     print(f"all_tokens: {all_tokens}")
                #     print(f"flaw_tokens_encoded: {flaw_tokens_encoded}")
                    
                for i in range(len(flaw_tokens_encoded)):
                    encoded_flaw = ''.join(flaw_tokens_encoded[i])
                    encoded_all = ''.join(all_tokens)
                    if encoded_flaw in encoded_all:
                        verified_flaw_lines.append(flaw_tokens_encoded[i])
                        do_explaination = True
                        
                if do_explaination:
                    # calculate the explanation scores
                    if reasoning_method == 'attention':
                        input_ids = input_ids.to(args.device)
                        loss, pred, last_hidden_state, attentions = model(input_ids=input_ids, labels=labels, output_attentions=True)
                        attentions = attentions[0][0]
                        attention = None
                        for i in range(len(attentions)):
                            layer_attention = attentions[i]
                            # summerize the values of each token dot other tokens
                            layer_attention = sum(layer_attention)
                            if attention is None:
                                attention = layer_attention
                            else:
                                attention += layer_attention
                        # clean att score for <s> and </s>
                        attention = clean_special_token_values(attention, padding=True)
                        # attention_score should be 1D tensor with seq length representing each token's attention_score value
                        word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)
                        all_lines_score, flaw_line_indices = get_all_lines_score(word_att_scores, verified_flaw_lines)
                        
                        explaination_path_dir = '/ssd2/wqq/work4/LineProbe/src/resource/firstStepModels/unixcoder/explaination'
                        explaination_path = os.path.join(explaination_path_dir, reasoning_method + '_explaination.csv')
                        # print(f"explaination_path: {explaination_path}")
                        
                        words = str([item[0] for item in word_att_scores])
                        scores = str([item[1].cpu().detach().numpy().item() for item in word_att_scores])
                        lines_score = str([item.cpu().detach().numpy().item() for item in all_lines_score])
                        
                        if not os.path.isfile(explaination_path):
                            logger.info(f"Save Explaination to {explaination_path}")
                            df = pd.DataFrame({'index':index, 'lines_score':[lines_score], 'flaw_line_index':[flaw_line_indices]})
                            df.to_csv(explaination_path, index=False)
                        else:
                            df = pd.read_csv(explaination_path)
                            new_data = pd.DataFrame({'index':index, 'lines_score':[lines_score], 'flaw_line_index':[flaw_line_indices]}, index=[0])
                            df = df._append(new_data, ignore_index=True)
                            df.to_csv(explaination_path, index=False)
                                                
                    elif reasoning_method == "deeplift" or \
                        reasoning_method == "deeplift_shap" or \
                        reasoning_method == "gradient_shap" or \
                        reasoning_method == "saliency":
                            
                        # send data to device
                        input_ids = input_ids.to(args.device)
                        if isinstance(model, torch.nn.DataParallel):
                            model = model.module
                        input_embed = model.encoder.roberta.embeddings(input_ids).to(args.device)
                        if reasoning_method == "deeplift":
                            baselines = torch.zeros(1, 512, 768, requires_grad=True).to(args.device)
                            reasoning_model = DeepLift(model)
                        elif reasoning_method == "deeplift_shap":
                            baselines = torch.zeros(16, 512, 768, requires_grad=True).to(args.device)
                            reasoning_model = DeepLiftShap(model)
                        elif reasoning_method == "gradient_shap":
                            baselines = torch.zeros(16, 512, 768, requires_grad=True).to(args.device)
                            reasoning_model = GradientShap(model)
                        elif reasoning_method == "saliency":
                            reasoning_model = Saliency(model)
                            
                        # attributions -> [1, 512, 768]
                        if reasoning_method == "saliency":
                            attributions = reasoning_model.attribute(input_embed, target=1)
                        else:
                            # print(f"input_embed: {input_embed.shape} baselines: {baselines.shape}")
                            attributions = reasoning_model.attribute(input_embed, baselines=baselines, target=1)
                        attributions_sum = summarize_attributions(attributions)        
                        attr_scores = attributions_sum.tolist()
                        # each token should have one score
                        assert len(all_tokens) == len(attr_scores)
                        # store tokens and attr scores together in a list of tuple [(token, attr_score)]
                        word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attr_scores)
                        # remove <s>, </s>, <unk>, <pad>
                        word_att_scores = clean_word_att_scores(word_att_scores=word_att_scores)
                        all_lines_score, flaw_line_indices = get_all_lines_score(word_att_scores, verified_flaw_lines)
                            
                        explaination_path_dir = '/ssd2/wqq/work4/LineProbe/src/resource/firstStepModels/unixcoder/explaination'
                        explaination_path = os.path.join(explaination_path_dir, reasoning_method + '_explaination.csv')
                        # print(f"explaination_path: {explaination_path}")
                        
                        if not os.path.isfile(explaination_path):
                            logger.info(f"Save Explaination to {explaination_path}")
                            df = pd.DataFrame({'index':index, 'lines_score':[all_lines_score], 'flaw_line_index':[flaw_line_indices]})
                            df.to_csv(explaination_path, index=False)
                        else:
                            df = pd.read_csv(explaination_path)
                            new_data = pd.DataFrame({'index':index, 'lines_score':[all_lines_score], 'flaw_line_index':[flaw_line_indices]}, index=[0])
                            df = df._append(new_data, ignore_index=True)
                            df.to_csv(explaination_path, index=False)
                
                index += 1

def generate_result_df(logits, y_trues, y_preds, args):
    file_path = args.test_data_file
    if 'csv' in file_path:
        df = pd.read_csv(file_path)
    if 'jsonl' in file_path:
        df = pd.read_json(file_path, orient="records", lines=True)
    all_num_lines = []
    all_processed_func = df["src"].tolist()
    for func in all_processed_func:
        all_num_lines.append(get_num_lines(func))
    flaw_line_index = df["flaw_line_index"].tolist()
    all_num_flaw_lines = []
    total_flaw_lines = 0
    for index in flaw_line_index:
        if isinstance(index, str):
            index = index.split(",")
            num_flaw_lines = len(index)
            total_flaw_lines += num_flaw_lines
        else:
            num_flaw_lines = 0
        all_num_flaw_lines.append(num_flaw_lines)
    assert len(logits) == len(y_trues) == len(y_preds) == len(all_num_flaw_lines)
    return pd.DataFrame({"logits": logits, "y_trues": y_trues, "y_preds": y_preds, 
                         "index": list(range(len(logits))), "num_flaw_lines": all_num_flaw_lines, "num_lines": all_num_lines, 
                         "flaw_line": df["flaw_line"], "src": df["src"]})

def write_raw_preds_csv(args, y_preds):
    df = pd.read_csv(args.test_data_file)
    df["raw_preds"] = y_preds
    df.to_csv("./results/raw_preds.csv", index=False)

def get_num_lines(func):
    func = func.split("\n")
    func = [line for line in func if len(line) > 0]
    return len(func)

def get_line_statistics(result_df):
    total_lines = sum(result_df["num_lines"].tolist())
    total_flaw_lines = sum(result_df["num_flaw_lines"].tolist())
    return total_lines, total_flaw_lines

def rank_lines(all_lines_score_with_label, is_attention, ascending_ranking):
    # flatten the list
    all_lines_score_with_label = [line for lines in all_lines_score_with_label for line in lines]
    if is_attention:
        all_scores = [line[0].item() for line in all_lines_score_with_label]
    else:
        all_scores = [line[0] for line in all_lines_score_with_label]
    all_labels = [line[1] for line in all_lines_score_with_label]
    rank_df = pd.DataFrame({"score": all_scores, "label": all_labels})
    rank_df = rank_dataframe(rank_df, "score", ascending_ranking)
    return len(rank_df), rank_df

def rank_dataframe(df, rank_by: str, ascending: bool):
    df = df.sort_values(by=[rank_by], ascending=ascending)
    df = df.reset_index(drop=True)
    return df


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def get_all_flaw_lines(flaw_lines: str, flaw_line_seperator: str) -> list:
    if isinstance(flaw_lines, str):
        flaw_lines = flaw_lines.strip(flaw_line_seperator)
        flaw_lines = flaw_lines.split(flaw_line_seperator)
        flaw_lines = [line.strip() for line in flaw_lines]
    else:
        flaw_lines = []
    return flaw_lines

def encode_all_lines(all_lines: list, tokenizer) -> list:
    encoded = []
    for line in all_lines:
        encoded.append(encode_one_line(line=line, tokenizer=tokenizer))
    return encoded

def encode_one_line(line, tokenizer):
    # add "@ " at the beginning to ensure the encoding consistency, i.e., previous -> previous, not previous > pre + vious
    code_tokens = tokenizer.tokenize("@ " + line)
    return [token.replace("Ġ", "") for token in code_tokens if token != "@"]

def clean_special_token_values(all_values, padding=False):
    # special token in the beginning of the seq 
    all_values[0] = 0
    if padding:
        # get the last non-zero value which represents the att score for </s> token
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        # special token in the end of the seq 
        all_values[-1] = 0
    return all_values

def get_word_att_scores(all_tokens: list, att_scores: list) -> list:
    word_att_scores = []
    for i in range(len(all_tokens)):
        token, att_score = all_tokens[i], att_scores[i]
        word_att_scores.append([token, att_score])
    return word_att_scores

def get_all_lines_score(word_att_scores: list, verified_flaw_lines: list):
    verified_flaw_lines = [''.join(l) for l in verified_flaw_lines]
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]
    # to return
    all_lines_score = []
    score_sum = 0
    line_idx = 0
    flaw_line_indices = []
    line = ""
    for i in range(len(word_att_scores)):
        # summerize if meet line separator or the last token
        if ((word_att_scores[i][0] in separator) or (i == (len(word_att_scores) - 1))) and score_sum != 0:
            score_sum += word_att_scores[i][1]
            all_lines_score.append(score_sum)
            is_flaw_line = False
            for l in verified_flaw_lines:
                if l == line:
                    is_flaw_line = True
            if is_flaw_line:
                flaw_line_indices.append(line_idx)
            line = ""
            score_sum = 0
            line_idx += 1
        # else accumulate score
        elif word_att_scores[i][0] not in separator:
            line += word_att_scores[i][0]
            score_sum += word_att_scores[i][1]
    return all_lines_score, flaw_line_indices

def create_ref_input_ids(input_ids, ref_token_id, sep_token_id, cls_token_id):
    seq_length = input_ids.size(1)
    ref_input_ids = [cls_token_id] + [ref_token_id] * (seq_length-2) + [sep_token_id]
    return torch.tensor([ref_input_ids])

def clean_word_att_scores(word_att_scores: list) -> list:
    to_be_cleaned = ['<s>', '</s>', '<unk>', '<pad>']
    cleaned = []
    for word_attr_score in word_att_scores:
        if word_attr_score[0] not in to_be_cleaned:
            cleaned.append(word_attr_score)
    return cleaned

def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--do_explaination", action='store_true',
                        help="Whether to do explaination.")
    parser.add_argument("--explaination_method", default=None, type=str)

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ") 
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    
    args = parser.parse_args()
    
    # Setup CUDA, GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count() - 1
    args.n_gpu = 4
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)    
    model = UnixCoderForVulnerabilityDetection(model, config, tokenizer, args)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    return results

if __name__ == "__main__":
    # log
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    file = logging.FileHandler('info.log')
    file.setLevel(level=logging.INFO)
    formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
    file.setFormatter(formatter)
    logger.addHandler(file)
    
    main()
