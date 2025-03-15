import re
from .utils import remove_comments_and_docstrings_c
from itertools import chain

def line2tokens(line, lang='c'):
    if lang == 'solidity':
        pattern = r'\b(?:\d+\s)?\w+\b|==|\+\+|--|\+=|-=|&&|\|\||\!=|<=|>=|\*\*|"[^"]*"|\'[^\']*\'|[^\w\s;]'
    elif lang == 'c':
        pattern = r'\d+\.\d+|\.\d+|\b(?:\d+\s)?\w+\b|==|\+\+|--|\+=|-=|->|&&|\|=|\|\||\!=|<=|>=|#\S*|\;|>>|<<|-\d+[a-zA-Z_]*|"[^"]*"|[^\w\s;]'
    elif lang == 'java':
        pattern = r'\b(?:\d+\s)?\w+\b|==|\+\+|--|\+=|-=|\!\=|&&|\|\||\!=|<=|>=|"[^"]*"|\'[^\']*\'|[^\w\s]'
    result = re.findall(pattern, line)

    return result

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

def token2line(term, row_idx, test_set, lang, show=False):
    code = test_set[row_idx]['src'].strip()
    ground_tokens = eval(term['token_sequence'])
    token_sum = []
    code = remove_comments_and_docstrings_c(code)
    for line in [line for line in code.split('\n')]:
        line = line2tokens(line, lang)
        token_sum.append(line)
    
    if row_idx == 2652:
        token_sum[2].append('\n')
        
    if row_idx == 2840:
        # print(f"token_sum: {token_sum}")
        token_sum[9].remove('\\')
        token_sum[9].remove('0')
        token_sum[9].insert(6, '\\0')
        
    if row_idx == 3455:
        # print(f"token_sum: {token_sum}")
        token_sum[1].insert(0, '')
        token_sum[17].remove('*')
        token_sum[17].remove('=')
        token_sum[17].insert(1, '*=')
        
    if row_idx == 4144:
        # print(f"token_sum: {token_sum}")
        token_sum[14].remove('&')
        token_sum[14].remove('=')
        token_sum[14].insert(3, '&=')
        
        token_sum[15].remove('&')
        token_sum[15].remove('=')
        token_sum[15].insert(3, '&=')
        
    if row_idx == 4148:
        # print(f"token_sum: {token_sum}")
        token_sum[3].insert(2, ' ')
        
    if row_idx == 213:
        for nn in [2,3,4,5]:
            token_sum[nn].remove('1.0')
            token_sum[nn].insert(6, '1.0f')
            token_sum[nn].remove('0.0')
            token_sum[nn].insert(10, '0.0f')
    
    if list(chain(*token_sum)) == ground_tokens:
        return True, token_sum
    else:
        # if show:
        #     
        #     logger.info(f"different: {row_idx}")
        #     print(f"row_idx: {row_idx}")
        #     print(f"code: {code}")
        #     print(f"a = {ground_tokens}")
        #     print(f"b = {list(chain(*token_sum))}")
        #     sys.exit()
        #     with open('/ssd2/wqq/work4/LineProbe/debug.txt', '+a') as f:
        #         f.write(f"row_idx: {row_idx}\n")
        #         f.write(f"a = {ground_tokens}\n")
        #         f.write(f"b = {list(chain(*token_sum))}\n")
        #     
        index = 0
        for i in range(len(token_sum)):
            for j in range(len(token_sum[i])):
                if token_sum[i][j] != ground_tokens[index]:
                    # 
                    token_sum[i].insert(j, "")
                    if list(chain(*token_sum)) == ground_tokens:
                        return True, token_sum
                    break
                else:
                    index += 1
        return False, [line for line in code.split('\n')]

def load_result_path(args, model_type: str, name: str, strategy: str):
    import os, sys

    if strategy in ['plain', 'frequency', 'ast']:
        if model_type == 'codebert':
            probe_result_path = os.path.join(args.probe_saved_path, f"codebert_c_128_model_{name}_probe_42/multiset.csv")
            test_result_path = f'VulProbe/src/resource/firstStepModels/BERT/test_result_{name}.csv'
        elif model_type == 'unixcoder':
            probe_result_path = os.path.join(args.probe_saved_path, f"unixcoder_c_128_model_{name}_probe_42/multiset.csv")
            test_result_path = f'VulProbe/src/resource/firstStepModels/unixcoder/test_result_{name}.csv'
        elif model_type == 'bilstm':
            probe_result_path = os.path.join(args.probe_saved_path, f"bilstm_c_128_model_{name}_probe_42/multiset.csv")
            test_result_path = f'VulProbe/src/resource/firstStepModels/BiLSTM/test_result_{name}.csv'
        elif model_type == 'mlp':
            probe_result_path = os.path.join(args.probe_saved_path, f"mlp_c_128_model_{name}_probe_42/multiset.csv")
            test_result_path = f'VulProbe/src/resource/firstStepModels/MLP/test_result_{name}.csv'
        elif model_type == 'codet5':
            probe_result_path = os.path.join(args.probe_saved_path, f"codet5_c_128_1215/multiset.csv")
            test_result_path = 'VulProbe/src/resource/firstStepModels/codet5/test_result.csv'
        elif model_type == 'graphcodebert':
            probe_result_path = os.path.join(args.probe_saved_path, f"graphcodebert_c_128_model_0/multiset.csv")
            test_result_path = f'VulProbe/src/resource/firstStepModels/graphcodebert/test_result_{name}.csv'
        else:
            sys.exit(f"No such first step model: {model_type}.")
                        
    if not os.path.exists(probe_result_path):
        sys.exit(f"Probe result {probe_result_path} not exists")
    if not os.path.exists(test_result_path):
        sys.exit(f"Test result {test_result_path} not exists")

            
    return probe_result_path, test_result_path

def load_customized_ast(args, model_type: str, name: str):
    import os, sys
    if model_type == 'codebert':
        customized_ast_path = os.path.join(args.probe_saved_path, f"codebert_c_128_model_{name}_probe_42")
    elif model_type == 'unixcoder':
        customized_ast_path = os.path.join(args.probe_saved_path, f"unixcoder_c_128_model_{name}_probe_42")
    elif model_type == 'bilstm':
        customized_ast_path = os.path.join(args.probe_saved_path, f"bilstm_c_128_model_{name}_probe_42")
    elif model_type == 'mlp':
        customized_ast_path = os.path.join(args.probe_saved_path, f"mlp_c_128_model_{name}_probe_42")
    else:
        sys.exit(f"No such first step model: {model_type}.")

    return os.path.join(customized_ast_path, 'customized_ast')
