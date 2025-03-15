import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_mean


# def get_embeddings(all_inputs, all_attentions, model, layer, model_type):
#     if model_type == 't5':
#         with torch.no_grad():
#             embs = model(input_ids=all_inputs, attention_mask=all_attentions)[1][layer][:, 1:, :]
#     else:
#         with torch.no_grad():
#             embs = model(input_ids=all_inputs, attention_mask=all_attentions)[2][layer][:, 1:, :]
#     return embs

def get_embeddings(all_inputs, model, layer, first_step_model_type, func=None, args=None):
    if first_step_model_type == 'textCNN':
        _, hidden_state = model(all_inputs)
        return hidden_state
    elif first_step_model_type == 'mlp':
        _, hidden_state = model(all_inputs)
        return hidden_state
    elif first_step_model_type == 'bilstm':
        input_embeds = model.codebert(all_inputs)[0].float()
        _, hidden_state = model(input_embeds)
        return hidden_state
    elif first_step_model_type == 'LineVul':
        with torch.no_grad():
            hidden_state = model(input_ids=all_inputs, layer=layer)
        return hidden_state
    elif first_step_model_type == 'codebert':
        with torch.no_grad():
            prob, outputs, _ = model(input_ids=all_inputs, output_attentions=True)
        return outputs
    elif first_step_model_type == 'graphcodebert':
        from tree_sitter import Language, Parser
        C_LANGUAGE = Language('VulProbe/src/resource/grammars/languages.so', 'c')
        tokenizer = model.tokenizer
        block_size = all_inputs.size(1)
        with torch.no_grad():
            # prob, outputs, _ = model(input_ids=all_inputs, output_attentions=True)
            parser = Parser()
            parser.set_language(C_LANGUAGE)
            type_embeds = []
            source_embeds = []
            
            for f in func:
                tree = parser.parse(bytes(f, "utf8"))
                root_node = tree.root_node
                type_list = []
                walk(root_node, type_list)
                type_list = type_list[:block_size-2]
                type_list = [tokenizer.cls_token] + type_list + [tokenizer.sep_token]
                type_ids = tokenizer.convert_tokens_to_ids(type_list)
                padding_length = block_size - len(type_ids)
                type_ids += [tokenizer.pad_token_id] * padding_length
                type_ids_tensor = torch.tensor([type_ids]).to(args.device)
                type_embed = model.encoder.roberta.embeddings(type_ids_tensor)
                type_embeds.append(type_embed)
                
                # source
                code_tokens = tokenizer.tokenize(str(f))[:block_size-2]
                source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                padding_length = block_size - len(source_ids)
                source_ids += [tokenizer.pad_token_id] * padding_length
                source_ids_tensor = torch.tensor([source_ids]).to(args.device)
                source_embed = model.encoder.roberta.embeddings(source_ids_tensor)
                source_embeds.append(source_embed)
            
            type_embeds = torch.cat(type_embeds, dim=0)
            source_embeds = torch.cat(source_embeds, dim=0)
            input_embed = type_embeds + source_embeds
            outputs = input_embed.cpu()
            
        return outputs
    elif first_step_model_type == 'codet5':
        with torch.no_grad():
            prob, outputs, _ = model(input_ids=all_inputs, output_attentions=True)
        return outputs
    elif first_step_model_type == 'unixcoder':
        with torch.no_grad():
            prob, outputs, _ = model(input_ids=all_inputs, output_attentions=True)
        return outputs
    else:
        raise ValueError(f"Unknown first_step_model_type: {first_step_model_type}")
    
def walk(node, type_list):
    if node.child_count == 0:
        parent = node.parent
        if parent:
            # print(f"Leaf: {node.type} -> Parent: {parent.type}")
            type_list.append(str(node.type) + '#' + str(parent.type))
        else:
            # print(f"Leaf: {node.type} -> No Parent")
            type_list.append(str(node.type) + '#' + 'root')
    else:
        for child in node.children:
            walk(child, type_list)

def align_function(embs, align):
    seq = []
    embs = embs[:, :-1, :]
    # print("\n")
    # print(f"embs, align: {len(embs), len(align)}")
    # print("\n")
    # print(f"embs, align: {len(embs[0]), len(align[0])}")
    # print("\n")
    # print(f"embs, align: {len(embs[0][0])}")
    # print("\n")
    for j, emb in enumerate(embs):
        seq.append(scatter_mean(emb, align[j], dim=0))
    # remove the last token since it corresponds to <\s> or padding to much the lens
    return pad_sequence(seq, batch_first=True)[:, :-1, :]
