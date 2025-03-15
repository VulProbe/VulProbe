import json
import logging
import os
import pathlib
import random
import re
import shutil
from collections import Counter

from tqdm import tqdm
from tree_sitter import Language, Parser

from .binary_tree import ast2binary, tree_to_distance
from .code2ast import code2ast, get_tokens_ast

logger = logging.getLogger(__name__)

LANGUAGES = (
    'c',
    'java'
)


# print(f"location: {os.getcwd()}")
C_LANGUAGE = Language('./src/resource/grammars/languages.so', 'c')
JAVA_LANGUAGE = Language('./src/resource/grammars/languages.so', 'java')
# C_LANGUAGE = Language('/ssd2/wqq/work4/for_exhibition/ast_probe/grammars/languages.so', 'c')
# JAVA_LANGUAGE = Language('/ssd2/wqq/work4/for_exhibition/ast_probe/grammars/languages.so', 'java')

C_PARSER = Parser()
C_PARSER.set_language(C_LANGUAGE)
JAVA_PARSER = Parser()
JAVA_PARSER.set_language(JAVA_LANGUAGE)

def convert_sample_to_features(term, parser, lang):
    if type(term) == str:
        G, pre_code = code2ast(term, parser, lang)
        binary_ast = ast2binary(G)
        d, c, _, u = tree_to_distance(binary_ast, 0)
        code_tokens = get_tokens_ast(G, pre_code)

        return {
            'd': d,
            'c': c,
            'u': u,
            'num_tokens': len(code_tokens),
            'code_tokens': code_tokens
        }
    else:
        code = term['src']
        
        G, pre_code = code2ast(code, parser, lang)
        binary_ast = ast2binary(G)
        d, c, _, u = tree_to_distance(binary_ast, 0)
        code_tokens = get_tokens_ast(G, pre_code)
        return {
            'd': d,
            'c': c,
            'u': u,
            'num_tokens': len(code_tokens),
            'code_tokens': code_tokens,
            'label': term['label'],
            'flaw_line_index': term['flaw_line_index'],
            'src': code
        }


def get_non_terminals_labels(train_set_labels, valid_set_labels, test_set_labels):
    all_labels = [label for seq in train_set_labels for label in seq] + \
                 [label for seq in valid_set_labels for label in seq] + \
                 [label for seq in test_set_labels for label in seq]
    # use a Counter to constantly get the same order in the labels
    ct = Counter(all_labels)
    labels_to_ids = {}
    for i, label in enumerate(ct):
        labels_to_ids[label] = i
    return labels_to_ids


def convert_to_ids(c, column_name, labels_to_ids):
    labels_ids = []
    for label in c:
        labels_ids.append(labels_to_ids[label])
    return {column_name: labels_ids}
