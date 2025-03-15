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

logger = logging.getLogger(__name__)

def rank_lines(lines_result, test_set):
    tp_list = test_result[test_result.apply(lambda x: x['label'] == 1 and x['pred'] == True, axis=1)].index.tolist()
    fp_list = test_result[test_result.apply(lambda x: x['label'] == 0 and x['pred'] == True, axis=1)].index.tolist()

    func_idx = []
    line_score = []
    is_vul = []
    flaw_line_idxs = []
    for row_idx, term in lines_result.iterrows():
        if term['index'] in tp_list or term['index'] in fp_list:
            flaw_line_idx = test_set['flaw_line_index'][row_idx]
            flaw_line_idx = revision(args, row_idx, flaw_line_idx)
            for idx, score in enumerate(eval(term['lines_score'])):
                # line_statistic[idx] = score
                func_idx.append(term['index'])
                line_score.append(score)
                flaw_line_idxs.append(str(flaw_line_idx))
                if idx in [int(x) for x in flaw_line_idx]:
                    is_vul.append(1)
                else:
                    is_vul.append(0)
                    
    df = pd.DataFrame({'func_idx': func_idx, 'line_score': line_score, 'is_vul': is_vul, 'flaw_line_index': flaw_line_idxs})
    df.to_csv('VulProbe/src/study/no_rank.csv', index=False)
    df = df.sort_values(by=['line_score'], ascending=False).reset_index(drop=True)
    df.to_csv('VulProbe/src/study/ranked.csv', index=False)
    
    return df

def get_recall(lines_result, test_set, top_k_loc, flaw_line_total_num, line_total_num):
    ranked_df = rank_lines(lines_result, test_set)
    
    target_line_num = line_total_num * top_k_loc
    checked_line_num = 0
    correct_predicted_line_num = 0
    for idx, term in ranked_df.iterrows():
        checked_line_num += 1
        if term['is_vul'] == 1:
            correct_predicted_line_num += 1
        if checked_line_num >= target_line_num:
            return round(correct_predicted_line_num / flaw_line_total_num, 6)
    return 0


def get_effort(lines_result, test_set, top_k_loc, flaw_line_total_num, line_total_num):
    ranked_df = rank_lines(lines_result, test_set)
    
    target_flaw_line_num = flaw_line_total_num * top_k_loc
    checked_line_num = 0
    correct_predicted_line_num = 0
    for idx, term in ranked_df.iterrows():
        checked_line_num += 1
        if term['is_vul'] == 1:
            correct_predicted_line_num += 1
        if correct_predicted_line_num >= target_flaw_line_num:
            return round(checked_line_num / line_total_num, 6)
    return 1
    

def check(ground_flaw_lines, pred_flaw_lines):
    # to check all ground_flaw_lines are in the pred_flaw_lines (length=10)
    # print(f'ground_flaw_lines: {ground_flaw_lines}\npred_flaw_lines: {pred_flaw_lines}\n')
    for term in ground_flaw_lines:
        if term in pred_flaw_lines:
            return True
    return False
    
    # all match
    # for term in ground_flaw_lines:
    #     if term in pred_flaw_lines:
    #         continue
    #     else:
    #         return False
    # return True
    
def remove_comments_and_docstrings_c(string):
    """Source: https://stackoverflow.com/questions/2319019/using-regex-to-remove-comments-from-source-files"""
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string

    return regex.sub(_replacer, string)
def revision(args, idx, flaw_line_idx):
    if args.explain_method in ['gpt']:
        flaw_line_idx = flaw_line_idx
    else:
        label = test_set['label'][idx]
        func = test_set['func_before'][idx]
        flaw_line = test_set['flaw_line'][idx]
        flaw_line = [x.strip() for x in flaw_line.split('/~/')]
        if label == 1 and len(flaw_line) > 0:
            if len(flaw_line) == 1 and flaw_line[0] == '':
                flaw_line_idx = []
            else:
                func = remove_comments_and_docstrings_c(func)
                flaw_line_index_list = []
                flaw_line_index_listid = 0
                for lineid, line in enumerate([x.strip() for x in func.split('\n')]):
                    if flaw_line[flaw_line_index_listid] == line:
                        flaw_line_index_list.append(lineid)
                        flaw_line_index_listid += 1
                        if flaw_line_index_listid >= len(flaw_line):
                            break
                flaw_line_idx = flaw_line_index_list
            # with open('VulProbe/aaa/see.txt', 'a+') as f:
            #     f.write("\n______________________________\n")
            #     f.write(str(idx) + '\n')
            #     f.write(test_set['func_before'][idx])
            #     f.write('\n______________________________\n')
            #     f.write(str(test_set['flaw_line_index'][idx]) + '\n')
            #     f.write(str(flaw_line_idx) + '\n')
            #     f.write('\n______________________________\n')
            #     f.write(func)
                
    return flaw_line_idx


if __name__ == '__main__':
    parser = HfArgumentParser(ProgramArguments)
    args = parser.parse_args()
        
    logger.info('Load test dataset from local file.')
    args.dataset_path = os.path.join(args.dataset_path, args.lang)
    test_set = pd.read_json(os.path.join(args.dataset_path, 'test.jsonl'), lines=True)
    
    top_k = args.top_k
    line_total_num = 0
    flaw_line_total_num = 0
    for idx, term in test_set.iterrows():
        code = term['src']
        line_total_num += len([line for line in code.split('\n') if len(line) > 0])
        flaw_line_idx = term['flaw_line_index']
        if ',' in flaw_line_idx:
            flaw_line_total_num += len(eval(flaw_line_idx))
            test_set.at[idx, 'flaw_line_index'] = list(eval(flaw_line_idx))
        else:
            if len(flaw_line_idx) > 0:
                flaw_line_total_num += 1
                test_set.at[idx, 'flaw_line_index'] = [eval(flaw_line_idx)]

    logger.info(f"Evaluate Attention Localization Method")
    
    hit = 0 

    # ifa
    line_num_to_check = list()
    
    l_ifa_line_num_to_check = list()
    
    if args.first_step_model_type == 'codebert':
        test_result = pd.read_csv(f'VulProbe/src/resource/firstStepModels/BERT/test_result_{args.name}.csv')
        if args.explain_method in ['deeplift_shap', 'gradient_shap', 'saliency', 'attention']:
            lines_result_path = f"VulProbe/src/resource/firstStepModels/BERT/explaination_{args.name}/{args.explain_method}_explaination.csv"
    
    if args.first_step_model_type == 'graphcodebert':
        test_result = pd.read_csv(f'VulProbe/src/resource/firstStepModels/graphcodebert/test_result_{args.name}.csv')
        if args.explain_method in ['deeplift', 'gradient_shap', 'saliency', 'attention', 'deeplift_shap']:
            lines_result_path = f"VulProbe/src/resource/firstStepModels/graphcodebert/explaination_{args.name}/{args.explain_method}_explaination.csv"
    
    if args.first_step_model_type == 'unixcoder':
        test_result = pd.read_csv(f'VulProbe/src/resource/firstStepModels/unixcoder/test_result_{args.name}.csv')
    
    if args.first_step_model_type == 'codet5':
        test_result = pd.read_csv('VulProbe/src/resource/firstStepModels/codet5/test_result.csv')
        if args.explain_method in ['lig', 'deeplift', 'gradient_shap', 'saliency', 'attention']:
            lines_result_path = f"VulProbe/src/resource/firstStepModels/codet5/explaination/{args.explain_method}_explaination.csv"
        
    tmp_idx = args.name
    if args.first_step_model_type == 'LineVul' and args.explain_method == 'linevul':
        test_result = pd.read_csv(f'LineVul/linevul/test_result_{tmp_idx}.csv')
        lines_result_path = f'LineVul/loc_result/line_att_scores_df_model_{tmp_idx}.csv'
        
    if args.first_step_model_type == 'bilstm':
        test_result = pd.read_csv(f'VulProbe/src/resource/firstStepModels/BiLSTM/test_result_{args.name}.csv')
        if args.explain_method in ['lig', 'deeplift', 'gradient_shap', 'saliency']:
            lines_result_path = f"VulProbe/src/resource/firstStepModels/BiLSTM/explaination/{args.explain_method}_explaination.csv"
        
    if args.first_step_model_type == 'mlp':
        test_result = pd.read_csv(f'VulProbe/src/resource/firstStepModels/MLP/test_result_{args.name}.csv')
        if args.explain_method in ['lig', 'deeplift', 'gradient_shap', 'saliency']:
            lines_result_path = f"VulProbe/src/resource/firstStepModels/MLP/explaination/{args.explain_method}_explaination.csv"
        
    tmp_idx = args.name # 0, 1, 2, 3, 4
    if args.first_step_model_type == 'linevd':
        test_result = pd.read_csv(f'VulProbe/results/baseline/linevd_score_{tmp_idx}.csv')
        lines_result_path = f'VulProbe/results/baseline/linevd_score_{tmp_idx}.csv'
        
    if args.first_step_model_type == 'qwen':
        test_result = pd.read_csv('VulProbe/llm/qwen/result2.csv')
        lines_result_path = 'VulProbe/llm/qwen/result2.csv'
        
    if args.first_step_model_type == 'gpt':
        test_result = pd.read_csv(f'VulProbe/llm/gpt/result_{args.name}.csv')
        lines_result_path = f'VulProbe/llm/gpt/result_{args.name}.csv'
        
    if args.explain_method == 'probe':
        lines_result_path = f'VulProbe/results/explaination/{args.first_step_model_type}_{args.strategy}_{args.explain_method}_explaination_{args.name}.csv'
    
    
    lines_result = pd.read_csv(lines_result_path)
    print(f"Load line probe result from: {lines_result_path} | shape: {lines_result.shape}")
    
    if 'lig' in lines_result_path:
        lines_result['lines_score'] = lines_result['lines_score'].apply(lambda x: x.replace('nan', '0'))    
        
    tp_list = test_result[test_result.apply(lambda x: x['label'] == 1 and x['pred'] == True, axis=1)].index.tolist()
    fp_list = test_result[test_result.apply(lambda x: x['label'] == 0 and x['pred'] == True, axis=1)].index.tolist()
    tn_list = test_result[test_result.apply(lambda x: x['label'] == 0 and x['pred'] == False, axis=1)].index.tolist()
    fn_list = test_result[test_result.apply(lambda x: x['label'] == 1 and x['pred'] == False, axis=1)].index.tolist()
    print(f"tp: {len(tp_list)} | fp: {len(fp_list)} | tn: {len(tn_list)} | fn: {len(fn_list)}")
    
    # record the hit entry
    hit_list = list()
    not_hit_list = list()
    
    # top_k_accuracy
    # save pred top_k
    lines_result['pred'] = None
    # for row_idx, term in tqdm(lines_result.iterrows(), total=len(lines_result)):
    for row_idx, term in lines_result.iterrows():
        if term['index'] in tn_list or term['index'] in fn_list:
            continue
        
        flaw_line_idx = test_set['flaw_line_index'][row_idx]
        flaw_line_idx = revision(args, row_idx, flaw_line_idx)
        
        line_statistic = {}
        for idx, score in enumerate(eval(term['lines_score'])):
            line_statistic[idx] = score
         
        ground_flaw_lines = [int(x) for x in flaw_line_idx]
        sorted_lines = sorted(line_statistic.items(), key=lambda item: item[1], reverse=True)
        lines_index = [x[0] for x in sorted_lines]
        lines_result.at[row_idx, 'pred'] = lines_index
        pred_flaw_lines = lines_index[:top_k]
        if check(ground_flaw_lines, pred_flaw_lines):
            # if term['index'] in tp_list:
            hit += 1
            hit_list.append(term['index'])
        else:
            not_hit_list.append(term['index'])
        # # for check
        # else:
        #     print(f"ground_flaw_lines: {ground_flaw_lines}\n \
        #             top_k_list: {top_k_list}\n \
        #             pred_flaw_lines: {pred_flaw_lines}\n")
            
    test_data = pd.read_json('VulProbe/src/resource/dataset/c/test.jsonl', lines=True)
    # ifa
    # for row_idx, term in tqdm(lines_result.iterrows(), total=len(lines_result)):
    for row_idx, term in lines_result.iterrows():
        flaw_line_idx = test_set['flaw_line_index'][row_idx]
        flaw_line_idx = revision(args, row_idx, flaw_line_idx)
        
        line_statistic = {}
        for idx, score in enumerate(eval(term['lines_score'])):
            line_statistic[idx] = score
            
        if term['index'] not in tp_list:
            if term['index'] in fn_list:
                ground_flaw_lines = [int(x) for x in flaw_line_idx]
                for ground_flaw_line in ground_flaw_lines:
                    line_num_to_check.insert(0, len(line_statistic))
                l_ifa_line_num_to_check.insert(0, len(line_statistic))
            continue
        
        try:
            for ground_flaw_line in [int(x) for x in flaw_line_idx]:
                line_num_to_check.insert(0, 0)
                for pred_flaw_line in [x[0] for x in sorted(line_statistic.items(), key=lambda item: item[1], reverse=True)]:
                    if ground_flaw_line == pred_flaw_line:
                        break
                    else:
                        line_num_to_check[0] += 1
        except Exception as e:
            traceback.print_exc()
            continue
        
        pred = term['pred']
        if not isinstance(pred, list):
            pred = eval(pred)
        l_ifa_line_num_to_check.insert(0, 0)
        for rank, pred_line in enumerate(pred):
            if pred_line in [int(x) for x in flaw_line_idx]:
                l_ifa_line_num_to_check[0] = rank
                break
            if rank == len(pred) - 1:
                l_ifa_line_num_to_check[0] = len(pred)
    
    flaw_line_total_num = flaw_line_total_num
        
    # ifa
    # print(f"sum(line_num_to_check): {sum(line_num_to_check)} len(line_num_to_check): {len(line_num_to_check)}")
    avg_ifa = sum(line_num_to_check) / len(line_num_to_check)
    median_ifa = statistics.median(line_num_to_check)
    # print(f'sum(l_ifa_line_num_to_check): {sum(l_ifa_line_num_to_check)} len(l_ifa_line_num_to_check): {len(l_ifa_line_num_to_check)}')
    linevul_ifa = sum(l_ifa_line_num_to_check) / len(l_ifa_line_num_to_check)
    
    # print(f'line_total_num: {line_total_num} | flaw_line_total_num: {flaw_line_total_num}')
    
    # Recall@1%Effort
    # rEffort = get_recall_1_effort_avg(lines_result, test_set, args.lang, flaw_line_total_num, line_total_num)
    rEffort = get_recall(lines_result, test_set, 0.01, flaw_line_total_num, line_total_num)
    
    # Effort@20%Recall
    # eRecall = get_effort_20_recall_avg(lines_result, test_set, args.lang, flaw_line_total_num, line_total_num)
    eRecall = get_effort(lines_result, test_set, 0.2, flaw_line_total_num, line_total_num)
    
    
    print(f"hit_num: {hit} | "
        f"top_{top_k}_accuracy: {round(hit / (len(tp_list) + len(fn_list)), 4)} | "
        f"IFA: {round(linevul_ifa, 2)} | "
        f"meanRank: {round(avg_ifa, 2)} | "
        f"Recall@1%Effort: {rEffort} | "
        f"Effort@20%Recall: {eRecall}")
    
    with open('VulProbe/mannual check/codebert_hit.txt', 'a+') as f:
        f.write(f"top_{top_k} hit list {len(hit_list)}:\n {hit_list}\n")
        f.write(f"top_{top_k} not hit list {len(not_hit_list)}:\n {not_hit_list}\n")
