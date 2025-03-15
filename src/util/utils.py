import re
import os
import pickle
import matplotlib.pyplot as plt

def picture_ast_generation(model_type, result_path, lang):
    precision = []
    recall = []
    f1 = []
    if model_type == 'LineVul':
        for layer in range(1, 13):
            file_name = f"linevul_{lang}_{layer}_128/metrics.log"
            path = os.path.join(result_path, file_name)
            with open(path, 'rb') as f:
                result = pickle.load(f)
            precision.append(result['test_precision'])
            recall.append(result['test_recall'])
            f1.append(result['test_f1'])

        x = list(range(1, 13))
        plt.xticks(x)
        plt.plot(x, precision, label='Precision')
        plt.plot(x, recall, label='Recall')
        plt.plot(x, f1, label='F1')
        plt.title('Probe results')
        plt.xlabel('Layer')
        plt.ylabel('Probe results')
        plt.legend()
        plt.savefig(os.path.join(f"./results/ast-probe-figure/linevul_{lang}.png"))

def remove_comments_and_docstrings_java_js(string):
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


def match_tokenized_to_untokenized_roberta(untokenized_sent, tokenizer):
    tokenized = []
    mapping = {}
    cont = 0
    for j, t in enumerate(untokenized_sent):
        if j == 0:
            temp = [k for k in tokenizer.tokenize(t) if k != 'Ġ']
            tokenized.append(temp)
            mapping[j] = [f for f in range(cont, len(temp) + cont)]
            cont = cont + len(temp)
        else:
            temp = [k for k in tokenizer.tokenize(' ' + t) if k != 'Ġ']
            tokenized.append(temp)
            mapping[j] = [f for f in range(cont, len(temp) + cont)]
            cont = cont + len(temp)
    flat_tokenized = [item for sublist in tokenized for item in sublist]
    return flat_tokenized, mapping

def get_frequency(multiset, test_result):
    ground_nodetypes = []
    for idx, item in multiset.iterrows():
        ground = eval(item['ground_multiset'])
        for node in ground:
            ground_nodetypes.append(node.split('-')[0])

    inter_nodetypes = []
    for idx, item in multiset.iterrows():
        intersection = eval(item['intersection'])
        for node in intersection:
            inter_nodetypes.append(node.split('-')[0])

    import json
    x_labels = list(set(inter_nodetypes))
    ground_y = [ground_nodetypes.count(i) for i in x_labels]
    inter_y = [inter_nodetypes.count(i) for i in x_labels]

    # delta_y = [abs(ground_y[i] - inter_y[i]) / ground_y[i] for i in range(len(x_labels))]
    delta_y = [inter_y[i] / ground_y[i] for i in range(len(x_labels))]
    # delta_y = [inter_y[i] for i in range(len(x_labels))]

    sum = 0
    for i in delta_y:
        sum += i
    type_score = {}
    for i in range(len(delta_y)):
        # type_score[x_labels[i]] = delta_y[i] / sum
        type_score[x_labels[i]] = sum / delta_y[i]
        
    return type_score
        
    

def get_ast_type_frequency(p_ast_preorders, f_ast_preorders):
    ground_nodetypes = []
    for sub_trees in f_ast_preorders:
        for sub_tree in sub_trees:
            if '-' in sub_tree:
                ground_nodetypes.append(sub_tree.split('-')[0])
    
    inter_nodetypes = []
    for idx, subtrees in enumerate(p_ast_preorders):
        for sub_tree in subtrees:
            if sub_tree in f_ast_preorders[idx]:
                if '-' in sub_tree:
                    inter_nodetypes.append(sub_tree.split('-')[0])
        
    x_labels = list(set(inter_nodetypes))
    ground_y = [ground_nodetypes.count(i) for i in x_labels]
    inter_y = [inter_nodetypes.count(i) for i in x_labels]

    # delta_y = [abs(ground_y[i] - inter_y[i]) / ground_y[i] for i in range(len(x_labels))]
    # delta_y = [inter_y[i] / ground_y[i] for i in range(len(x_labels))]
    # delta_y = [inter_y[i] for i in range(len(x_labels))]
    
    delta_y = [abs(ground_y[i] - inter_y[i]) for i in range(len(x_labels))]
    min_delta_y = min(delta_y)
    max_delta_y = max(delta_y)
    
    type_score = {}
    for i in range(len(x_labels)):
        delta_y = abs(ground_y[i] - inter_y[i])
        type_score[x_labels[i]] = (max_delta_y - delta_y) / (max_delta_y - min_delta_y)
    return type_score

    sum = 0
    for i in delta_y:
        sum += i
    type_score = {}
    for i in range(len(delta_y)):
        # type_score[x_labels[i]] = delta_y[i] / sum
        type_score[x_labels[i]] = sum / delta_y[i]
        
    # print(type_score)
        
    return type_score
