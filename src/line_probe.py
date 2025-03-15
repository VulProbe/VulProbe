import os
import sys
import argparse
import logging
import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn

from tree_sitter import Parser
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, T5Config, T5ForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader

from util import C_LANGUAGE, JAVA_LANGUAGE, get_non_terminals_labels, convert_to_ids, collator_fn
from util.data_loading import get_non_terminals_labels, convert_to_ids, convert_sample_to_features
from util.binary_tree import distance_to_tree, remove_empty_nodes, extend_complex_nodes, get_precision_recall_f1, add_unary
from probe import ParserProbe, ParserLoss, get_embeddings, align_function

from resource.firstStepModels.BiLSTM.bilstm import codebert_lstm, BiLSTM_Config
from resource.firstStepModels.MLP.mlp import codebert_MLP, MLP_Config
from resource.firstStepModels.BERT.bert_model import CodeBert
from resource.firstStepModels.unixcoder.unixcoder_model import UnixCoderForVulnerabilityDetection

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

def train(args: argparse.Namespace):
    logger.info('-' * 100)
    logger.info('Start training Probe.')
    logger.info('-' * 100)
    
    # select the parser
    parser = Parser()
    if args.lang == 'c':
        parser.set_language(C_LANGUAGE)
    elif args.lang == 'java':
        parser.set_language(JAVA_LANGUAGE)
    
    pretrained_model_path = "microsoft/codebert-base"
    # pretrained model config
    config = RobertaConfig.from_pretrained(pretrained_model_path)
    config.num_labels = 1
    
    # pretrained model tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
    
    # configure the first step model
    logger.info('Configure the first step model')        
    if args.first_step_model_type == 'mlp':
        mlp_config = MLP_Config
        lmodel = codebert_MLP(mlp_config)
        mlp_path = os.path.join("./src/resource/firstStepModels/MLP", args.first_step_model)
        lmodel.load_state_dict(torch.load(mlp_path, map_location=args.device))
        for param in lmodel.parameters():
            param.requires_grad = False
        lmodel = lmodel.to(args.device)
        criterion = nn.CrossEntropyLoss()

    if args.first_step_model_type == 'bilstm':
        bilstm_config = BiLSTM_Config
        lmodel = codebert_lstm(bilstm_config.hidden_dim, bilstm_config.output_size, bilstm_config.n_layers, bilstm_config.bidirectional, bilstm_config.drop_prob)
        bilstm_path = "./src/resource/firstStepModels/BiLSTM/codebert_bilstm_classifier.pth"
        lmodel.load_state_dict(torch.load(bilstm_path))
        
        for param in lmodel.parameters():
            param.requires_grad = False
        lmodel = lmodel.to(args.device)
        criterion = nn.CrossEntropyLoss()
        
        
    if args.first_step_model_type == 'codebert':
        # Load model
        lmodel = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config, ignore_mismatched_sizes=True)
        lmodel = CodeBert(lmodel, config, tokenizer, args)
        weights = os.path.join('./src/resource/firstStepModels/BERT/checkpoint-best-f1', args.first_step_model)
        lmodel.load_state_dict(torch.load(weights, map_location=args.device), strict=False)
        lmodel = lmodel.to(args.device)
        
    if args.first_step_model_type == 'unixcoder':
        # Load model
        pretrained_model_path = "microsoft/unixcoder-base-nine"
        config = RobertaConfig.from_pretrained(pretrained_model_path)
        config.num_labels = 1
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
        
        lmodel = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config, ignore_mismatched_sizes=True)
        lmodel = UnixCoderForVulnerabilityDetection(lmodel, config, tokenizer, args)
        weights = os.path.join('./src/resource/firstStepModels/unixcoder/checkpoint-best-f1', args.first_step_model)
        lmodel.load_state_dict(torch.load(weights, map_location=args.device), strict=False)
        lmodel = lmodel.to(args.device)
        

    # load the dataset
    logger.info('Load dataset from local file.')
    data_files = {'train': os.path.join(args.dataset_path, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_path, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_path, 'test.jsonl')}
    train_set = load_dataset('json', data_files=data_files, split='train')
    valid_set = load_dataset('json', data_files=data_files, split='valid')
    test_set = load_dataset('json', data_files=data_files, split='test')
    train_set = train_set.map(lambda e: convert_sample_to_features(e, parser, args.lang))
    valid_set = valid_set.map(lambda e: convert_sample_to_features(e, parser, args.lang))
    test_set = test_set.map(lambda e: convert_sample_to_features(e, parser, args.lang))
    
    # generate the mapping for tokens' labels
    labels_file_path = os.path.join(args.dataset_path, 'labels.pkl')
    if not os.path.exists(labels_file_path):
        # convert each non-terminal labels to its id
        labels_to_ids_c = get_non_terminals_labels(train_set['c'], valid_set['c'], test_set['c'])
        ids_to_labels_c = {x: y for y, x in labels_to_ids_c.items()}
        labels_to_ids_u = get_non_terminals_labels(train_set['u'], valid_set['u'], test_set['u'])
        ids_to_labels_u = {x: y for y, x in labels_to_ids_u.items()}
        with open(labels_file_path, 'wb') as f:
            pickle.dump({
                'labels_to_ids_c': labels_to_ids_c, 'ids_to_labels_c': ids_to_labels_c,
                'labels_to_ids_u': labels_to_ids_u, 'ids_to_labels_u': ids_to_labels_u
            }, f)
    else:
        with open(labels_file_path, 'rb') as f:
            data = pickle.load(f)
            labels_to_ids_c = data['labels_to_ids_c']
            ids_to_labels_c = data['ids_to_labels_c']
            labels_to_ids_u = data['labels_to_ids_u']
            ids_to_labels_u = data['ids_to_labels_u']
    train_set = train_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    valid_set = valid_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    test_set = test_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))

    train_set = train_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))
    valid_set = valid_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))
    test_set = test_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))
    
    # generate dataloader
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)
    valid_dataloader = DataLoader(dataset=valid_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)
    
    # load Probe Model
    probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c),
        number_labels_u=len(labels_to_ids_u)).to(args.device)
    
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    criterion = ParserLoss(loss='rank')
    
    # start training the Probe Model
    logger.info(f"{os.path.join(args.output_path, 'probe_model.bin')}")
    if os.path.exists(os.path.join(args.output_path, 'probe_model.bin')):
        logger.info('model trained')
    else:
        probe_model.train()
        lmodel.eval()
        # if args.first_step_model_type == 'bilstm':
        #     lmodel.train()
        best_eval_loss = float('inf')
        metrics = {'training_loss': [], 'validation_loss': [], 'test_precision': None, 'test_recall': None, 'test_f1': None}
        patience_count = 0
        
        for epoch in range(1, args.epochs + 1):
            training_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc='[training batch]', bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
                token_sequences, flaw_line_idxes, labels, all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment, codes = batch
                if args.first_step_model_type == 'bilstm':
                    # Adjust hidden state size based on the current batch size
                    current_batch_size = all_inputs.size(0)
                all_inputs = all_inputs.to(args.device)
                if args.first_step_model_type == 'graphcodebert':
                    embds = get_embeddings(all_inputs, lmodel, args.layer, args.first_step_model_type, codes, args)
                else:
                    embds = get_embeddings(all_inputs, lmodel, args.layer, args.first_step_model_type)
                # print(f'aligment: {alignment.shape} embds: {embds.shape} all_inputs: {all_inputs.size(1)}')
                # sys.exit(0)
                embds = align_function(embds.to(args.device), alignment.to(args.device))

                d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
                loss = criterion(
                    d_pred=d_pred.to(args.device),
                    scores_c=scores_c.to(args.device),
                    scores_u=scores_u.to(args.device),
                    d_real=ds.to(args.device),
                    c_real=cs.to(args.device),
                    u_real=us.to(args.device),
                    length_batch=batch_len_tokens.to(args.device))

                reg = args.orthogonal_reg * (torch.norm(torch.matmul(torch.transpose(probe_model.proj, 0, 1), probe_model.proj)
                                                        - torch.eye(args.rank).to(args.device)) ** 2)
                loss += reg
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                training_loss += loss.item()
                
            training_loss = training_loss / len(train_dataloader)
            eval_loss, _, _, _ = eval(valid_dataloader, probe_model, lmodel, criterion, args)
            scheduler.step(eval_loss)
            logger.info(f'[epoch {epoch}] train loss: {round(training_loss, 4)}, validation loss: {round(eval_loss, 4)}')
            metrics['training_loss'].append(round(training_loss, 4))
            metrics['validation_loss'].append(round(eval_loss, 4))
            print(f"eval_loss: {eval_loss}")
            if eval_loss < best_eval_loss:
                logger.info('-' * 100)
                logger.info('Saving model checkpoint')
                logger.info('-' * 100)
                output_path = os.path.join(args.output_path, f'probe_model.bin')
                torch.save(probe_model.state_dict(), output_path)
                logger.info(f'Probe model saved: {output_path}')
                patience_count = 0
                best_eval_loss = eval_loss
            else:
                patience_count += 1
            if patience_count == args.patience:
                logger.info('Stopping training loop (out of patience).')
                break
        
    if os.path.exists(os.path.join(args.output_path, 'metrics.log')):
        logger.info('model tested')
    else:
        logger.info("Start testing the Probe.")
        logger.info('Loading test set.')
        test_dataloader = DataLoader(dataset=test_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                    num_workers=8)
        
        logger.info('Load the best model.')
        checkpoint = torch.load(os.path.join(args.output_path, 'probe_model.bin'))
        probe_model.load_state_dict(checkpoint)
        
        logger.info('Test.')
        eval_precision, eval_recall, eval_f1_score = ast_probe_test(test_dataloader, probe_model, lmodel,
                                                                        ids_to_labels_c, ids_to_labels_u, args)
        metrics = {'training_loss': [], 'validation_loss': [], 'test_precision': None, 'test_recall': None, 'test_f1': None}
        metrics['test_precision'] = round(eval_precision, 4)
        metrics['test_recall'] = round(eval_recall, 4)
        metrics['test_f1'] = round(eval_f1_score, 4)
        logger.info(f'test precision: {round(eval_precision, 4)} | test recall: {round(eval_recall, 4)} '
                    f'| test F1 score: {round(eval_f1_score, 4)}')

        logger.info('-' * 100)
        logger.info('Saving metrics.')
        with open(os.path.join(args.output_path, 'metrics.log'), 'wb') as f:
            pickle.dump(metrics, f)
            
def eval(test_dataloader, probe_model, lmodel, criterion, args):
    probe_model.eval()
    eval_loss = 0.0
    total_hits_c = 0
    total_c = 0
    total_hits_u = 0
    total_u = 0
    total_hits_d = 0
    total_d = 0
        
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,
                                          desc='[test batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            token_sequences, flaw_line_idxes, labels, all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment, codes = batch
            ds = ds.to(args.device)
            cs = cs.to(args.device)
            us = us.to(args.device)

            if args.first_step_model_type == 'bilstm':
                    # Adjust hidden state size based on the current batch size
                    current_batch_size = all_inputs.size(0)
            if args.first_step_model_type == 'graphcodebert':
                embds = get_embeddings(all_inputs, lmodel, args.layer, args.first_step_model_type, codes, args)
            else:
                embds = get_embeddings(all_inputs, lmodel, args.layer, args.first_step_model_type)
            embds = align_function(embds.to(args.device), alignment.to(args.device))

            d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
            loss = criterion(
                d_pred=d_pred.to(args.device),
                scores_c=scores_c.to(args.device),
                scores_u=scores_u.to(args.device),
                d_real=ds.to(args.device),
                c_real=cs.to(args.device),
                u_real=us.to(args.device),
                length_batch=batch_len_tokens.to(args.device))
            eval_loss += loss.item()

            # compute the classes c and u
            # scores_c /= probe_model.vectors_c.norm(p=2, dim=0)
            # scores_u /= probe_model.vectors_u.norm(p=2, dim=0)
            scores_c = torch.argmax(scores_c, dim=2)
            scores_u = torch.argmax(scores_u, dim=2)

            batch_len_tokens = batch_len_tokens.to(args.device)
            lens_d = (batch_len_tokens - 1).to(args.device)
            max_len_d = torch.max(lens_d)
            mask_c = torch.arange(max_len_d, device=args.device)[None, :] < lens_d[:, None]
            mask_u = torch.arange(max_len_d + 1, device=args.device)[None, :] < batch_len_tokens[:, None]

            scores_c = torch.masked_select(scores_c, mask_c)
            scores_u = torch.masked_select(scores_u, mask_u)
            cs = torch.masked_select(cs, mask_c)
            us = torch.masked_select(us, mask_u)

            hits_c = (scores_c == cs).sum().item()
            hits_u = (scores_u == us).sum().item()

            total_hits_u += hits_u
            total_hits_c += hits_c
            total_c += mask_c.sum().item()
            total_u += mask_u.sum().item()

            hits_d, total_d_current = compute_hits_d(d_pred, ds, mask_c)
            total_hits_d += hits_d
            total_d += total_d_current

    acc_u = float(total_hits_u) / float(total_u)
    acc_c = float(total_hits_c) / float(total_c)
    acc_d = float(total_hits_d) / float(total_d)

    return (eval_loss / len(test_dataloader)), acc_c, acc_u, acc_d

def compute_hits_d(input, target, mask):
    diff = input[:, :, None] - input[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = mask[:, :, None] * mask[:, None, :] * target_diff
    loss = torch.relu(target_diff - diff)
    hits = (((loss * mask) == 0) * mask).sum().item()
    return hits, mask.sum().item()

def ast_probe_test(test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args):
    
    probe_model.eval()
    precisions, recalls, f1_scores = [], [], []
        
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,
                                          desc='[test batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            token_sequences, flaw_line_idxes, labels, all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment, codes = batch
            
            if args.first_step_model_type == 'graphcodebert':
                embds = get_embeddings(all_inputs, lmodel, args.layer, args.first_step_model_type, codes, args)
            else:
                embds = get_embeddings(all_inputs, lmodel, args.layer, args.first_step_model_type)
            embds = align_function(embds.to(args.device), alignment.to(args.device))

            d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
            scores_c = torch.argmax(scores_c, dim=2)
            scores_u = torch.argmax(scores_u, dim=2)

            probe_datasets = []
            for i, len_tokens in enumerate(batch_len_tokens):
                # vul info
                token_sequence = token_sequences[i]
                # flaw_line = flaw_lines[i]
                flaw_line_idx = flaw_line_idxes[i]
                label = labels[i]
                # probe info
                len_tokens = len_tokens.item()
                d_pred_current = d_pred[i, 0:len_tokens - 1].tolist()
                score_c_current = scores_c[i, 0:len_tokens - 1].tolist()
                score_u_current = scores_u[i, 0:len_tokens].tolist()
                ds_current = ds[i, 0:len_tokens - 1].tolist()
                cs_current = cs[i, 0:len_tokens - 1].tolist()
                us_current = us[i, 0:len_tokens].tolist()

                cs_labels = [ids_to_labels_c[c] for c in cs_current]
                us_labels = [ids_to_labels_u[c] for c in us_current]
                scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
                scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

                ground_truth_tree = distance_to_tree(ds_current, cs_labels, us_labels,
                                                     [str(i) for i in range(len_tokens)])
                ground_truth_tree = extend_complex_nodes(add_unary(remove_empty_nodes(ground_truth_tree)))

                pred_tree = distance_to_tree(d_pred_current, scores_c_labels, scores_u_labels,
                                             [str(i) for i in range(len_tokens)])
                pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))
                
                try:
                    p, r, f1_score, m_pred, m_true, intersection = get_precision_recall_f1(ground_truth_tree, pred_tree)
                except Exception as e:
                    p, r, f1_score = get_precision_recall_f1(ground_truth_tree, pred_tree)

                probe_dataset = {
                    'pred_multiset': m_pred, 
                    'ground_multiset': m_true, 
                    'intersection': intersection, 
                    'token_sequence': token_sequence, 
                    # 'flaw_line': flaw_line, 
                    'flaw_line_idx': flaw_line_idx, 
                    'label': label
                    }
                probe_datasets.append(probe_dataset)
                f1_scores.append(f1_score)
                precisions.append(p)
                recalls.append(r)
                
                
            #     continue
            
            probe_datasets = pd.DataFrame(probe_datasets)
            save_path = os.path.join(args.probe_saved_path, args.probe_name, 'multiset.csv')
            if os.path.exists(save_path):
                temp = pd.read_csv(save_path)
                temp = pd.concat([temp, probe_datasets])
                temp.to_csv(save_path, index=False)
            else:
                probe_datasets.to_csv(save_path, index=False)

    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)


def test(args: argparse.Namespace):
    logger.info('-' * 100)
    logger.info('Start testing Probe.')
    logger.info('-' * 100)
    
    # select the parser
    parser = Parser()
    if args.lang == 'c':
        parser.set_language(C_LANGUAGE)
    elif args.lang == 'java':
        parser.set_language(JAVA_LANGUAGE)
        
    pretrained_model_path = "microsoft/codebert-base"
    # pretrained model config
    config = RobertaConfig.from_pretrained(pretrained_model_path)
    config.num_labels = 1
    
    # pretrained model tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
            
    if args.first_step_model_type == 'codebert':
        # Load model
        pretrained_model_path = "microsoft/codebert-base"
        config = RobertaConfig.from_pretrained(pretrained_model_path)
        config.num_labels = 1
        
        # pretrained model tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
        
        lmodel = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config, ignore_mismatched_sizes=True)
        lmodel = CodeBert(lmodel, config, tokenizer, args)
        weights = os.path.join('./src/resource/firstStepModels/BERT/checkpoint-best-f1', args.first_step_model)
        lmodel.load_state_dict(torch.load(weights, map_location=args.device), strict=False)
        lmodel = lmodel.to(args.device)
        
    if args.first_step_model_type == 'unixcoder':
        # Load model
        pretrained_model_path = "microsoft/unixcoder-base-nine"
        config = RobertaConfig.from_pretrained(pretrained_model_path)
        config.num_labels = 1
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
        
        lmodel = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config, ignore_mismatched_sizes=True)
        lmodel = UnixCoderForVulnerabilityDetection(lmodel, config, tokenizer, args)
        weights = os.path.join('./src/resource/firstStepModels/unixcoder/checkpoint-best-f1', args.first_step_model)
        lmodel.load_state_dict(torch.load(weights, map_location=args.device), strict=False)
        lmodel = lmodel.to(args.device)
        
    if args.first_step_model_type == 'mlp':
        mlp_config = MLP_Config
        lmodel = codebert_MLP(mlp_config)
        args.first_step_model = args.first_step_model.split('.')[0] + '.pth'
        mlp_path = os.path.join("./src/resource/firstStepModels/MLP", args.first_step_model)
        lmodel.load_state_dict(torch.load(mlp_path, map_location=args.device))

        for param in lmodel.parameters():
            param.requires_grad = False
        lmodel = lmodel.to(args.device)

    if args.first_step_model_type == 'bilstm':
        bilstm_config = BiLSTM_Config
        lmodel = codebert_lstm(bilstm_config.hidden_dim, bilstm_config.output_size, bilstm_config.n_layers, bilstm_config.bidirectional, bilstm_config.drop_prob)
        bilstm_path = "./src/resource/firstStepModels/BiLSTM/codebert_bilstm_classifier.pth"
        lmodel.load_state_dict(torch.load(bilstm_path))

        for param in lmodel.parameters():
            param.requires_grad = False
        lmodel = lmodel.to(args.device)



    # load the dataset
    logger.info('Load dataset from local file.')
    data_files = {'train': os.path.join(args.dataset_path, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_path, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_path, 'test.jsonl')}
    train_set = load_dataset('json', data_files=data_files, split='train')
    valid_set = load_dataset('json', data_files=data_files, split='valid')
    test_set = load_dataset('json', data_files=data_files, split='test')
    train_set = train_set.map(lambda e: convert_sample_to_features(e, parser, args.lang))
    valid_set = valid_set.map(lambda e: convert_sample_to_features(e, parser, args.lang))
    test_set = test_set.map(lambda e: convert_sample_to_features(e, parser, args.lang))
    
    # generate the mapping for tokens' labels
    labels_file_path = os.path.join(args.dataset_path, 'labels.pkl')
    if not os.path.exists(labels_file_path):
        # convert each non-terminal labels to its id
        labels_to_ids_c = get_non_terminals_labels(train_set['c'], valid_set['c'], test_set['c'])
        ids_to_labels_c = {x: y for y, x in labels_to_ids_c.items()}
        labels_to_ids_u = get_non_terminals_labels(train_set['u'], valid_set['u'], test_set['u'])
        ids_to_labels_u = {x: y for y, x in labels_to_ids_u.items()}
        with open(labels_file_path, 'wb') as f:
            pickle.dump({
                'labels_to_ids_c': labels_to_ids_c, 'ids_to_labels_c': ids_to_labels_c,
                'labels_to_ids_u': labels_to_ids_u, 'ids_to_labels_u': ids_to_labels_u
            }, f)
    else:
        with open(labels_file_path, 'rb') as f:
            data = pickle.load(f)
            labels_to_ids_c = data['labels_to_ids_c']
            ids_to_labels_c = data['ids_to_labels_c']
            labels_to_ids_u = data['labels_to_ids_u']
            ids_to_labels_u = data['ids_to_labels_u']
            
    test_set = test_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    test_set = test_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))

    test_dataloader = DataLoader(dataset=test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                num_workers=8)
    
    logger.info('Load the probe.')
    probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c),
        number_labels_u=len(labels_to_ids_u)).to(args.device)
    checkpoint = torch.load(os.path.join(args.output_path, 'probe_model.bin'))
    probe_model.load_state_dict(checkpoint)

    logger.info('Test.')
    eval_precision, eval_recall, eval_f1_score = ast_probe_test_only_show(test_dataloader, probe_model, lmodel,
                                                                    ids_to_labels_c, ids_to_labels_u, args)
    metrics = {'training_loss': [], 'validation_loss': [], 'test_precision': None, 'test_recall': None, 'test_f1': None}
    metrics['test_precision'] = round(eval_precision, 4)
    metrics['test_recall'] = round(eval_recall, 4)
    metrics['test_f1'] = round(eval_f1_score, 4)
    logger.info(f'test precision: {round(eval_precision, 4)} | test recall: {round(eval_recall, 4)} '
                f'| test F1 score: {round(eval_f1_score, 4)}')

    logger.info('-' * 100)


def ast_probe_test_only_show(test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args):
    
    probe_model.eval()
    precisions, recalls, f1_scores = [], [], []
        
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,
                                          desc='[test batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            token_sequences, flaw_line_idxes, labels, all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment, codes = batch
            
            if args.first_step_model_type == 'graphcodebert':
                embds = get_embeddings(all_inputs, lmodel, args.layer, args.first_step_model_type, codes, args)
            else:
                all_inputs = all_inputs.to(args.device)
                lmodel = lmodel.to(args.device)
                embds = get_embeddings(all_inputs, lmodel, args.layer, args.first_step_model_type)
            embds = align_function(embds.to(args.device), alignment.to(args.device))

            d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
            scores_c = torch.argmax(scores_c, dim=2)
            scores_u = torch.argmax(scores_u, dim=2)

            probe_datasets = []
            for i, len_tokens in enumerate(batch_len_tokens):
                # vul info
                token_sequence = token_sequences[i]
                # flaw_line = flaw_lines[i]
                flaw_line_idx = flaw_line_idxes[i]
                label = labels[i]
                # probe info
                len_tokens = len_tokens.item()
                d_pred_current = d_pred[i, 0:len_tokens - 1].tolist()
                score_c_current = scores_c[i, 0:len_tokens - 1].tolist()
                score_u_current = scores_u[i, 0:len_tokens].tolist()
                ds_current = ds[i, 0:len_tokens - 1].tolist()
                cs_current = cs[i, 0:len_tokens - 1].tolist()
                us_current = us[i, 0:len_tokens].tolist()

                cs_labels = [ids_to_labels_c[c] for c in cs_current]
                us_labels = [ids_to_labels_u[c] for c in us_current]
                scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
                scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

                ground_truth_tree = distance_to_tree(ds_current, cs_labels, us_labels,
                                                     [str(i) for i in range(len_tokens)])
                ground_truth_tree = extend_complex_nodes(add_unary(remove_empty_nodes(ground_truth_tree)))

                pred_tree = distance_to_tree(d_pred_current, scores_c_labels, scores_u_labels,
                                             [str(i) for i in range(len_tokens)])
                pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))
                
                # Save the ground truth and predicted trees as nx.DiGraph
                output_path = os.path.join(args.output_path, 'customized_ast')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                ground_truth_tree_path = os.path.join(output_path, f'ground_truth_tree_{step}.pkl')
                pred_tree_path = os.path.join(output_path, f'pred_tree_{step}.pkl')

                import pickle
                
                try:
                    if not os.path.exists(ground_truth_tree_path):
                        customized_ast = customize_ast(ground_truth_tree)
                        with open(ground_truth_tree_path, 'wb') as f:
                            pickle.dump(customized_ast, f)
                        
                    if not os.path.exists(pred_tree_path):
                        customized_ast = customize_ast(pred_tree)
                        with open(pred_tree_path, 'wb') as f:
                            pickle.dump(customized_ast, f)
                    
                    p, r, f1_score, m_pred, m_true, intersection = get_precision_recall_f1(ground_truth_tree, pred_tree)
                except Exception as e:
                    p, r, f1_score = get_precision_recall_f1(ground_truth_tree, pred_tree)

                probe_dataset = {
                    'pred_multiset': m_pred, 
                    'ground_multiset': m_true, 
                    'intersection': intersection, 
                    'token_sequence': token_sequence, 
                    # 'flaw_line': flaw_line, 
                    'flaw_line_idx': flaw_line_idx, 
                    'label': label
                    }
                probe_datasets.append(probe_dataset)
                f1_scores.append(f1_score)
                precisions.append(p)
                recalls.append(r)
            
    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

def customize_ast(ast):
    # ast attributes
    # ast.nodes[node]['type'] : node type
    # ast.nodes[node]['depth'] : node depth
    # ast.nodes[node]['is_terminal'] : node is terminal or not
    # ast.nodes[node]['pre_order'] : node pre-order traversal
    # .out_degree(node) : number of successors
    # .in_degree(node) : number of predecessors
    # .successors(node) : successors
    # .predecessors(node) : predecessors
    
    from collections import deque
    # Find a root candidate (a node with no incoming edges)
    roots = [node for node, deg in ast.in_degree() if deg == 0]
    root = roots[0]
    ast.nodes[root]['depth'] = 0
    queue = deque([root])
    visited_nodes = {root}
    while queue:
        current = queue.popleft()
        current_depth = ast.nodes[current]['depth']
        children = list(ast.successors(current))
        
        ast.nodes[current]['is_terminal'] = (len(children) == 0)
        for child in children:
            if child not in visited_nodes:
                ast.nodes[child]['depth'] = current_depth + 1
                visited_nodes.add(child)
                queue.append(child)

    def build_po(node):
        # pre-order traversal: record current node's type then its descendants
        result = ast.nodes[node]['type']
        for child in ast.successors(node):
            result += '-' + build_po(child)
        return result

    # For every node in the graph, add a new attribute 'sub_tree'
    for node in ast.nodes:
        ast.nodes[node]['pre_order'] = build_po(node)
        
    return ast
