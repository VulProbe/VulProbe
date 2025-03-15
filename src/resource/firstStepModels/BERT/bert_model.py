import os
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(seed)
    
# Model Config
class BERT_Config:
    batch_size = 16
    hidden_dim = 768 
    lr = 2e-5
    drop_prob = 0.5
    sentence_max_size = 512
    epochs = 20
    use_cuda = USE_CUDA
    save_path = 'bert_classifier.pth'
    num_labels = 1
    pretrained_model_path = 'microsoft/codebert-base'
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path)
    
class RobertaClassificationHead(nn.Module):
    """Sub-Model for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class CodeBert(RobertaForSequenceClassification): 
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeBert, self).__init__(config=config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.loss_fn = CrossEntropyLoss()
        self.classifier = RobertaClassificationHead(config)
    
        
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            outputs = self.encoder.roberta(
                input_ids, 
                attention_mask=input_ids.ne(1), 
                output_hidden_states=True, 
                output_attentions=True
            ) # last_hidden_state (batch_size, sequence_length, hidden_size)
            # get last hidden state
            last_hidden_state = outputs.last_hidden_state
            # get attention
            attentions = outputs.attentions
            
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            
            if labels is not None:
                loss = self.loss_fn(logits, labels)
                return loss, prob, last_hidden_state, attentions
            else:
                return prob, last_hidden_state, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss = self.loss_fn(logits, labels)
                return loss, prob
            else:
                # print(f"prob.shape: {prob.shape}")
                return prob
        
        
        output = outputs[0]
        logits = output
        prob = torch.sigmoid(logits)
        # print(prob.size())
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.float().view(-1))
            # loss = torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            # loss = -loss.mean()
            return loss, prob, outputs.hidden_states[12]
        else:
            return prob, outputs.hidden_states[12]
      
        