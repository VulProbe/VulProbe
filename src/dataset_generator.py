import logging
import argparse
import random
import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
from transformers import AutoTokenizer
from tree_sitter import Parser
from datasets import load_dataset, Dataset

from util import C_LANGUAGE, JAVA_LANGUAGE
from util.code2ast import code2ast, get_tokens_ast, has_error
from util.utils import match_tokenized_to_untokenized_roberta

tokenizer_codebert = AutoTokenizer.from_pretrained('microsoft/codebert-base')

tokenizers = [tokenizer_codebert]


def filter_samples(code, max_length, lang, parser):
    try:
        G, code_pre = code2ast(code=code, parser=parser, lang=lang)
        assert nx.is_tree(nx.Graph(G))
        assert nx.is_connected(nx.Graph(G))
    except:
        return False
    if has_error(G):
        return False

    for tokenizer in tokenizers:
        t, _ = match_tokenized_to_untokenized_roberta(untokenized_sent=code_pre, tokenizer=tokenizer)
        if len(t) + 2 > max_length:
            return False
    return True

def code2token(code, lang):
    parser = Parser()
    if lang == 'c':
        parser.set_language(C_LANGUAGE)
    elif lang == 'java':
        parser.set_language(JAVA_LANGUAGE)
    G, pre_code = code2ast(code, parser, lang=lang)
    tokens = get_tokens_ast(G, pre_code)
    return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating the dataset for probing')
    parser.add_argument('--dataset_dir', default='./src/resource/dataset', help='Path to save the dataset')
    parser.add_argument('--lang', help='Language.', choices=['c', 'java'],
                        default='c')
    parser.add_argument('--max_code_length', help='Maximum code length.', default=512)
    parser.add_argument('--seed', help='seed.', type=int, default=123)
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    # seed everything
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
    dataset_path = os.path.join(args.dataset_dir, args.lang, 'dataset.jsonl')
    
    if not os.path.exists(dataset_path):
        filename = os.listdir(os.path.join(args.dataset_dir, args.lang))[0]
        csv_path = os.path.join(args.dataset_dir, args.lang, filename)
        df = pd.read_csv(csv_path)
        df.to_json(dataset_path, orient='records', lines=True)
    
    logger.info('Loading dataset.')
    if 'json' in dataset_path:
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    elif 'csv' in dataset_path:
        dataset = load_dataset('csv', data_files=dataset_path)
    
    # select the parser
    parser_lang = Parser()
    if args.lang == 'c':
        parser_lang.set_language(C_LANGUAGE)
    elif args.lang == 'java':
        parser_lang.set_language(JAVA_LANGUAGE)
    else:
        raise Exception("Use C or Java!")

    # filter dataset by code length
    logger.info('Filtering dataset.')
    dataset = dataset.filter(
        lambda e: filter_samples(e['src'], args.max_code_length, args.lang, parser_lang), num_proc=1)

    logger.info('Shuffling dataset.')
    dataset = dataset.shuffle(args.seed)
    
    # generate code_tokens from src
    logger.info('generate code_tokens')
    df_dataset = dataset.to_pandas()
    df_dataset['code_tokens'] = df_dataset['src'].apply(lambda x: code2token(x, args.lang))
    df_dataset = df_dataset[~((df_dataset['label'] == 1) & (df_dataset['flaw_line_index'].isna()))]
    dataset = Dataset.from_pandas(df_dataset)
    

    logger.info('Splitting dataset.')
    # train : valid : test = 8 : 1 : 1
    length = len(dataset)
    train_len = length * 8 // 10
    if (length - train_len) % 2 == 0:
        test_len, val_len = (length - train_len) // 2, (length - train_len) // 2
    else:
        train_len = train_len - 1
        test_len, val_len = (length - train_len) // 2, (length - train_len) // 2
    train_dataset = dataset.select(range(0, train_len))
    test_dataset = dataset.select(range(train_len, train_len + test_len))
    val_dataset = dataset.select(range(train_len + test_len, train_len + test_len + val_len))

    logger.info('Storing dataset.')
    train_dataset.to_json(os.path.join(args.dataset_dir, args.lang, 'train.jsonl'))
    test_dataset.to_json(os.path.join(args.dataset_dir, args.lang, 'test.jsonl'))
    val_dataset.to_json(os.path.join(args.dataset_dir, args.lang, 'valid.jsonl'))

    