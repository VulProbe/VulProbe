import os
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import time
from transformers import HfArgumentParser
from dataclasses import dataclass, field
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# model explaination
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency

@dataclass
class ProgramArguments:
    do_train: bool = field(default=False, metadata={'help': 'Train the MLP classifier.'})
    do_explain: bool = field(default=False, metadata={'help': 'Explain the MLP classifier.'})
    explaination_method: str = field(default='deeplift', metadata={'help': 'Explain the MLP classifier.'})
    save_path: str = field(default='codebert_mlp_classifier.pth', metadata={'help': 'The path to save the MLP classifier.'})
    seed: int = field(default=42, metadata={'help': 'Random seed.'})

USE_CUDA = torch.cuda.is_available()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
    
class MLP_Config:
    batch_size = 1
    input_dim = 768 
    output_size = 2
    lr = 2e-5
    drop_prob = 0.5
    epochs = 20
    use_cuda = USE_CUDA
    save_path = 'codebert_MLP_classifier.pth'
    num_labels = 2
    

class codebert_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        for param in self.codebert.parameters():
            param.requires_grad = False

        self.hidden_layer1 = nn.Linear(768, 384)
        kaiming_uniform_(self.hidden_layer1.weight, nonlinearity='relu')
        self.relu1 = nn.ReLU()
        
        self.hidden_layer2 = nn.Linear(384, 64)
        kaiming_uniform_(self.hidden_layer2.weight, nonlinearity='relu')
        
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64, 2)
        xavier_uniform_(self.fc.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.codebert(x, attention_mask=x.ne(1))[0].float()
        x = self.hidden_layer1(x)
        x = self.relu1(x)
        x = self.hidden_layer2(x)
        x = self.relu2(x)
        out_hidden = x
        x = x.mean(dim=1)
        x = self.fc(x)
        # return x 
        return x, out_hidden
        
def train_model(config, data_train):
    model = codebert_MLP(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if (config.use_cuda):
        model.cuda(args.device_id)
    
    best_f1 = 0
    for e in range(config.epochs):
        logger.info(f"Epoch {e}")
        # initialize hidden state
        counter = 0
        # batch loop
        for inputs, labels in tqdm(data_train, total=len(data_train)):
            model.train()
            counter += 1
            if (config.use_cuda):
                inputs, labels = inputs.cuda(args.device_id), labels.cuda(args.device_id)
            model.zero_grad()
            
            input_embeds = model.codebert(inputs, attention_mask=inputs.ne(1))[0].float()
            output = model(input_embeds)
            if config.batch_size > 1:
                output = output.squeeze()
            loss = criterion(output, labels.long())
            loss.backward()
            optimizer.step()
            
        # validation
        result = valid_model(config, model, valid_loader)
        if (result['valid_f1'] > best_f1):
            best_f1 = result['valid_f1']
            logger.info("  "+"*"*20)  
            logger.info("  Best f1:%s", round(best_f1,4))
            logger.info("  "+"*"*20)
            
            model_to_save = model.module if hasattr(model,'module') else model
            logger.info("Saving model checkpoint to %s", config.save_path)
            torch.save(model_to_save.state_dict(), config.save_path)

def valid_model(config, model, data_valid):
    if (config.use_cuda):
        model.cuda(args.device_id)
    criterion = nn.CrossEntropyLoss()
    valid_losses = []  # track loss
    
    correctT = 0
    total = 0
    classnum = 2
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))
    
    all_labels = []
    all_preds = []
    
    model.eval()
    # iterate over valid data
    for inputs, labels in data_valid:
        if (USE_CUDA):
            inputs, labels = inputs.cuda(args.device_id), labels.cuda(args.device_id)
            
        input_embeds = model.codebert(inputs, attention_mask=inputs.ne(1))[0].float()
        output = model(input_embeds)
        if config.batch_size > 1:
            output = output.squeeze()
        valid_loss = criterion(output, labels.long())
        valid_losses.append(valid_loss.item())
        _, pred = torch.max(output, 1)
        labels = Variable(labels)

        total += labels.size(0)
        correctT += pred.eq(labels.data).cpu().sum()
        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(output.size()).scatter_(1, labels.data.cpu().view(-1, 1).long(), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask * tar_mask
        acc_num += acc_mask.sum(0)
        
        # valid result save
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)   
    f1 = f1_score(all_labels, all_preds)             
    result = {
        "valid_accuracy": float(acc),
        "valid_recall": float(recall),
        "valid_precision": float(precision),
        "valid_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
        
    return result

# model explaination funcs
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
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions
def get_word_att_scores(all_tokens: list, att_scores: list) -> list:
    word_att_scores = []
    for i in range(len(all_tokens)):
        token, att_score = all_tokens[i], att_scores[i]
        word_att_scores.append([token, att_score])
    return word_att_scores
def clean_word_att_scores(word_att_scores: list) -> list:
    to_be_cleaned = ['<s>', '</s>', '<unk>', '<pad>']
    cleaned = []
    for word_attr_score in word_att_scores:
        if word_attr_score[0] not in to_be_cleaned:
            cleaned.append(word_attr_score)
    return cleaned
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
def test_model(config, data_test):
    logger.info("Start testing...")
    model = codebert_MLP(config)
    model.load_state_dict(torch.load(config.save_path))

    if (config.use_cuda):
        model.cuda(args.device_id)
    criterion = nn.CrossEntropyLoss()
    test_losses = []  # track loss
    
    num_correct = 0
    correctT = 0
    total = 0
    classnum = 2
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))
    
    
    all_labels = []
    all_preds = []
    
    model.eval()
    # iterate over test data
    for inputs, labels in data_test:
        if (USE_CUDA):
            inputs, labels = inputs.cuda(args.device_id), labels.cuda(args.device_id)
            
        input_embeds = model.codebert(inputs, attention_mask=inputs.ne(1))[0].float()
        output = model(input_embeds)
        if config.batch_size > 1:
            output = output.squeeze()
        test_loss = criterion(output, labels.long())
        test_losses.append(test_loss.item())
        _, pred = torch.max(output, 1)
        labels = Variable(labels)

        # test result save
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)   
    f1 = f1_score(all_labels, all_preds)             
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1)
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
        
    test_result_df = pd.DataFrame({
        "label": all_labels,
        "pred": all_preds
    })
    # logger.info(f"len(all_labels): {len(all_labels)}, len(all_preds): {len(all_preds)}")
    test_result_df.to_csv(f'test_result_{args.seed}.csv')
    
    if args.do_explain:
        methods = []
        if args.explaination_method == 'all':
            methods = ['lig', 'deeplift', 'gradient_shap', 'saliency']
        else:
            methods = [args.explaination_method]
            
        test_dataset = pd.read_json('VulProbe/src/resource/dataset/c/test.jsonl', orient="records", lines=True)
        result_df = test_dataset
        for reasoning_method in methods:
            logger.info(f"***** Explaining the Model with {reasoning_method} *****")
            index = 0
            index_list = []
            lines_score_list = []
            flaw_line_index_list = []
            for input_ids, labels in tqdm(data_test):
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
                
                for i in range(len(flaw_tokens_encoded)):
                    encoded_flaw = ''.join(flaw_tokens_encoded[i])
                    encoded_all = ''.join(all_tokens)
                    if encoded_flaw in encoded_all:
                        verified_flaw_lines.append(flaw_tokens_encoded[i])
                        do_explaination = True
                
                if not do_explaination:
                    index_list.append(index)
                    lines_score_list.append([])
                    flaw_line_index_list.append([])
                else:
                    if reasoning_method == "deeplift" or \
                        reasoning_method == "deeplift_shap" or \
                        reasoning_method == "gradient_shap" or \
                        reasoning_method == "saliency":
                            
                        # send data to device
                        input_ids = input_ids.to(args.device)
                        if isinstance(model, torch.nn.DataParallel):
                            model = model.module
                        input_embeds = model.codebert(inputs)[0].float()
                        if reasoning_method == "deeplift":
                            baselines = torch.zeros(1, 200, 768, requires_grad=True).to(args.device)
                            reasoning_model = DeepLift(model)
                        elif reasoning_method == "gradient_shap":
                            baselines = torch.zeros(16, 200, 768, requires_grad=True).to(args.device)
                            reasoning_model = GradientShap(model)
                        elif reasoning_method == "saliency":
                            reasoning_model = Saliency(model)
                            
                        # attributions -> [1, 512, 768]
                        if reasoning_method == "saliency":
                            attributions = reasoning_model.attribute(input_embeds, target=1)
                        else:
                            # print(type(input_embeds), type(baselines))
                            # print(f"input_embed: {input_embeds.shape} baselines: {baselines.shape}")
                            attributions = reasoning_model.attribute(input_embeds, baselines=baselines, target=1)

                        attributions_sum = summarize_attributions(attributions)        
                        attr_scores = attributions_sum.tolist()
                        # each token should have one score
                        assert len(all_tokens) == len(attr_scores)
                        # store tokens and attr scores together in a list of tuple [(token, attr_score)]
                        word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attr_scores)
                        # remove <s>, </s>, <unk>, <pad>
                        word_att_scores = clean_word_att_scores(word_att_scores=word_att_scores)
                        all_lines_score, flaw_line_indices = get_all_lines_score(word_att_scores, verified_flaw_lines)
                        
                        index_list.append(index)
                        lines_score_list.append(all_lines_score)
                        
                    index += 1
            
            explaination_path_dir = 'explaination'
            if not os.path.exists(explaination_path_dir):
                os.mkdir(explaination_path_dir)
            explaination_path = os.path.join(explaination_path_dir, reasoning_method + '_explaination.csv')
            logger.info(f"Save Explaination to {explaination_path}")
            df = pd.DataFrame({'index':index_list, 'lines_score':lines_score_list})
            df.to_csv(explaination_path, index=False)
def myDataProcess(dataFile):
    df = pd.read_json(dataFile, orient="records", lines=True)

    logger.info("load data successfully!")

    srcs = list(df['src'])
    labels = np.array(df['label'])

    return srcs, labels

if __name__ == '__main__':
    # log
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    
    # args
    parser = HfArgumentParser(ProgramArguments)
    args = parser.parse_args()
    logger.info(f"do train: {args.do_train}")
    
        # set seed
    set_seed(args.seed)
    
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
    
    start_time = time.time()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    args.device_id = 3
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 2
    args.device = device
    
    model_config = MLP_Config()
    model_config.save_path = args.save_path

    logger.info("***** Model config *****")
    for attr in dir(model_config):
        if not attr.startswith('__'):
            value = getattr(model_config, attr)
            logger.info(f'{attr} = {value}')
    logger.info("***** ************ *****")

    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    if args.do_train:
        logger.info("Start training...")
        train_srcs, train_labels = myDataProcess("VulProbe/src/resource/dataset/c/train.jsonl")
        valid_srcs, valid_labels = myDataProcess("VulProbe/src/resource/dataset/c/valid.jsonl")
        
        train_srcs = tokenizer(train_srcs,
                        padding=True,
                        truncation=True,
                        max_length=200,
                        return_tensors='pt')
        
        valid_srcs = tokenizer(valid_srcs,
                        padding=True,
                        truncation=True,
                        max_length=200,
                        return_tensors='pt')
        
        train_x = train_srcs['input_ids']
        train_y = torch.from_numpy(train_labels).float()
        valid_x = valid_srcs['input_ids']
        valid_y = torch.from_numpy(valid_labels).float()
        
        logger.info(f"Train Dataset Size: {len(train_x)}, Valid Dataset Size: {len(valid_x)}")
        X_train, X_valid, y_train, y_valid = train_x, valid_x, train_y, valid_y
        
        train_data = TensorDataset(X_train, y_train)
        valid_data = TensorDataset(X_valid, y_valid)
        
        train_loader = DataLoader(train_data,
                                    shuffle=True,
                                    batch_size=model_config.batch_size,
                                    drop_last=True)
        valid_loader = DataLoader(valid_data,
                                shuffle=True,
                                batch_size=model_config.batch_size,
                                drop_last=True)
        
        if (USE_CUDA):
            logger.info('Run on GPU.')
        else:
            logger.info('No GPU available, run on CPU.')

        train_model(model_config, train_loader)
        
        logger.info("Training finished.")
        end_time = time.time()
        total_time = end_time - start_time
        
        
    # test
    logger.info("Start testing...")
    
    test_srcs, test_labels = myDataProcess("VulProbe/src/resource/dataset/c/test.jsonl")    
    
    test_srcs = tokenizer(test_srcs,
                    padding=True,
                    truncation=True,
                    max_length=200,
                    return_tensors='pt')
        
    test_x = test_srcs['input_ids']
    test_y = torch.from_numpy(test_labels).float()

    logger.info(f"Test Dataset Size: {len(test_x)}")
    # X_train, X_test, y_train, y_test = train_x, test_x, train_y, test_y
    X_test, y_test = test_x, test_y

    test_data = TensorDataset(X_test, y_test)

    test_loader = DataLoader(test_data,
                                batch_size=model_config.batch_size,
                                drop_last=True)
        
    if (USE_CUDA):
        logger.info('Run on GPU.')
    else:
        logger.info('No GPU available, run on CPU.')

    test_model(model_config, test_loader)

        
