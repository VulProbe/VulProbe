import pandas as pd
import numpy as np
import os
import re
import sys
import pickle
import statistics
import logging
import json
from datetime import datetime
import traceback

from itertools import chain
from tqdm import tqdm
from tree_sitter import Parser, Language
from datasets import load_dataset
from datasets import Dataset
import matplotlib.pyplot as plt
from args import ProgramArguments
from transformers import HfArgumentParser

from util import C_LANGUAGE, JAVA_LANGUAGE
from util.probe_score import token2line, load_result_path, load_customized_ast
logger = logging.getLogger(__name__)




if __name__ == '__main__':
    parser = HfArgumentParser(ProgramArguments)
    args = parser.parse_args()
    
    # picture_ast_generation(args.first_step_model_type, args.probe_saved_path, args.lang)
            
    logger.info('Load test dataset from local file.')
    args.dataset_path = os.path.join(args.dataset_path, args.lang)
    data_files = {'train': os.path.join(args.dataset_path, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_path, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_path, 'test.jsonl')}
    test_set = load_dataset('json', data_files=data_files, split='test')
    
    model_type, name, strategy = args.first_step_model_type, args.name, args.strategy
    probe_result_path, test_result_path = load_result_path(args, model_type, name, strategy)
                            
    print(f"****** probe_result path: {probe_result_path} ******")
    probe_result = pd.read_csv(probe_result_path)
    for idx, term in probe_result.iterrows():
        if pd.isna(term['flaw_line_idx']):
            probe_result.at[idx, 'flaw_line_idx'] = []
        else:
            probe_result.at[idx, 'flaw_line_idx'] = eval('[' + term['flaw_line_idx'] + ']')

    logger.info(f"Select the *** {args.strategy} Strategy ***")
    
    bad_pattern_tokenization = 0
    bad_pattern_tokenization_line_num = 0 
    bad_pattern_tokenization_flaw_line_num = 0 
    nan_num = 0        
        
    print(f"****** test_result_path: {test_result_path} ******")
    test_result = pd.read_csv(test_result_path)
    tp_list = test_result[test_result.apply(lambda x: x['label'] == 1 and x['pred'] == True, axis=1)].index.tolist()
    fp_list = test_result[test_result.apply(lambda x: x['label'] == 0 and x['pred'] == True, axis=1)].index.tolist()
    tn_list = test_result[test_result.apply(lambda x: x['label'] == 0 and x['pred'] == False, axis=1)].index.tolist()
    fn_list = test_result[test_result.apply(lambda x: x['label'] == 1 and x['pred'] == False, axis=1)].index.tolist()
    print(f"tp: {len(tp_list)} | fp: {len(fp_list)} | tn: {len(tn_list)} | fn: {len(fn_list)} | total: {len(probe_result)}")
    
    
    if args.strategy == 'ast':
        logger.info('Load customized ASTs.')
        p_ast_list = []
        p_ast_preorders = []
        f_ast_list = []
        f_ast_preorders = []
        model_type, name = args.first_step_model_type, args.name
        customized_ast_path = load_customized_ast(args, model_type, name)
        print(f"****** customized_ast path: {customized_ast_path} ******")
        for ast_idx in range(len(os.listdir(customized_ast_path)) // 2):
            p_ast_name = f'pred_tree_{ast_idx}.pkl'
            f_ast_name = f'ground_truth_tree_{ast_idx}.pkl'
            with open(os.path.join(customized_ast_path, p_ast_name), 'rb') as f:
                pred_tree = pickle.load(f)
            with open(os.path.join(customized_ast_path, f_ast_name), 'rb') as f:
                ground_tree = pickle.load(f)
                
            p_ast_preorder = []
            for node in pred_tree.nodes:
                p_ast_preorder.append(pred_tree.nodes[node]['pre_order'])
            f_ast_preorder = []
            for node in ground_tree.nodes:
                f_ast_preorder.append(ground_tree.nodes[node]['pre_order'])
                
            p_ast_list.append(pred_tree)
            f_ast_list.append(ground_tree)
            p_ast_preorders.append(p_ast_preorder)
            f_ast_preorders.append(f_ast_preorder)
        # ast_data = pd.DataFrame({'pred_tree': p_ast_list, 'ground_tree': f_ast_list, 'pred_ast_preorder': p_ast_preorders, 'ground_ast_preorder': f_ast_preorders})
        
        # one = ast_data['pred_tree'][0]
        
        # for node in one.nodes:
        #     print(one.nodes[node])
        
    if args.strategy == 'frequency':
        from util.utils import get_frequency
        type_score = get_frequency(probe_result, test_result)
        
    if args.strategy == 'ast':
        from util.utils import get_ast_type_frequency
        type_score = get_ast_type_frequency(p_ast_preorders, f_ast_preorders)

    
    probe_result['statistic'] = None
    err_idx = set()
    for row_idx, term in tqdm(probe_result.iterrows(), total=len(probe_result)):
        token_num = len(eval(term['token_sequence'])) + 1
        statistic = {
            'sum': [0 for i in range(token_num)],
            'true': [0 for i in range(token_num)],
            'false': [0 for i in range(token_num)],
        }
        
        if args.strategy == 'ast':

            p_ast_preorder = p_ast_preorders[row_idx]
            f_ast_preorder = f_ast_preorders[row_idx]
            score = 1
            for node, ast_node in zip(eval(term['pred_multiset']), p_ast_preorder):
                if ast_node in f_ast_preorder:
                    sta = statistic['true']
                    node_type = node.split('-')[0]                    
                    if node_type in ['if_statement', 'else_clause', 'for_statement', 'while_statement', 'do_statement', 'switch_statement', 'case_statement', 'break_statement', 'continue_statement', 'goto_statement', 'labeled_statement']:
                        score = 1.5
                    for token_idx in node.split('-')[1:]:
                        token_idx = int(token_idx)
                        try:
                            sta[token_idx] += score
                            statistic['sum'][token_idx] += 1
                        except Exception as e:
                            if row_idx not in err_idx:
                                # print(f"row_idx: {row_idx}")
                                err_idx.add(row_idx)
                                # traceback.print_exc()
                            else:
                                break
                            
                else:
                    sta = statistic['false']
                    for token_idx in node.split('-')[1:]:
                        token_idx = int(token_idx)
                        try:
                            sta[token_idx] += 1
                            statistic['sum'][token_idx] += 1
                        except Exception as e:
                            break
            probe_result.at[row_idx, 'statistic'] = statistic
            
        elif args.strategy == 'plain':    
            score = 1
            for node in eval(term['pred_multiset']):
                if node in eval(term['ground_multiset']):
                    sta = statistic['true']
                    node_type = node.split('-')[0]                    
                    if node.split('-')[0] in ['if_statement', 'else_clause', 'for_statement', 'while_statement', 'do_statement', 'switch_statement', 'case_statement', 'break_statement', 'continue_statement', 'goto_statement', 'labeled_statement']:
                        score = 1.5
                    for token_idx in node.split('-')[1:]:
                        token_idx = int(token_idx)
                        try:
                            sta[token_idx] += score
                            statistic['sum'][token_idx] += 1
                        except Exception as e:
                            if row_idx not in err_idx:
                                # print(f"row_idx: {row_idx}")
                                err_idx.add(row_idx)
                                # traceback.print_exc()
                            else:
                                break
                            
                else:
                    sta = statistic['false']
                    for token_idx in node.split('-')[1:]:
                        token_idx = int(token_idx)
                        try:
                            sta[token_idx] += 1
                            statistic['sum'][token_idx] += 1
                        except Exception as e:
                            break
            probe_result.at[row_idx, 'statistic'] = statistic

                        
                else:
                    sta = statistic['false']
                    for token_idx in node.split('-')[1:]:
                        token_idx = int(token_idx)
                        try:
                            sta[token_idx] += 1
                            statistic['sum'][token_idx] += 1
                        except Exception as e:
                            break
            probe_result.at[row_idx, 'statistic'] = statistic
            
    for idx in err_idx:
        if idx in tp_list or idx in fn_list:
            print(f"err_idx: {idx}")
    
    # score aggregation
    probe_result['lines_score'] = None
    probe_result['index'] = None
    for row_idx, term in tqdm(probe_result.iterrows(), total=len(probe_result)):
        same, code_2_tokenized_lines = token2line(term, row_idx, test_set, args.lang)
        line_statistic = {}
        hit_idx, hit_statistic = 0, term['statistic']['true'][1:]
        
        if same:
            for line_idx, line in enumerate(code_2_tokenized_lines):
                line_statistic[line_idx] = 0
                for j in line:
                    line_statistic[line_idx] += hit_statistic[hit_idx]
                    hit_idx += 1
        else:
            if row_idx in tp_list:
                token2line(term, row_idx, test_set, args.lang, show=True)
                bad_pattern_tokenization += 1
                bad_pattern_tokenization_line_num += len(code_2_tokenized_lines)
                
        lines_score = [x for x in line_statistic.values()]
        if len(lines_score) > 0:
            lines_score[0] = 0
        probe_result.at[row_idx, 'lines_score'] = lines_score
        probe_result.at[row_idx, 'index'] = row_idx
    
    print(f"bad_pattern_tokenization: {bad_pattern_tokenization}")
    explain_result_save_path = f'VulProbe/results/explaination/{args.first_step_model_type}_{args.strategy}_probe_explaination_{args.name}.csv'
    print(f"save path: {explain_result_save_path}")
    probe_result.to_csv(explain_result_save_path)
    