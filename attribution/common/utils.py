
import torch
from torch.utils.data import Dataset, DataLoader
import copy
import re
import torch
from torch.cuda.amp import autocast
import numpy as np

class GRPO_dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=8192):
        """
        Assumes the dataset is a JSON list of records where each record is a dict with keys:
        "problem": the problem text,
        "solution": the solution text.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Concatenate problem and solution with a separator.
        text = item['text'] 
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            padding_side = 'right'
        )
        
        # Squeeze to remove the batch dimension.
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        # label id
        tokenized_prompt = self.tokenizer("", truncation=True, max_length=self.max_length)
        prompt_length = len(tokenized_prompt["input_ids"])
        response_length = len(input_ids) - prompt_length
        # labels = copy.copy(input_ids[prompt_length:])

        labels = copy.deepcopy(input_ids)
        if prompt_length !=0:
            labels[:prompt_length] = torch.tensor([-100] * prompt_length)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask,'response_length':response_length,'label':labels,'indices':torch.tensor(idx)}

def get_required_infos(dataloader,unq_tokens_ts,model,device):
    input_ids_all = []
    pb_diffs = []
    embedding_list = []
    attention_masks = []
    indices = []
    for _,batch in enumerate(dataloader):
        print('!!!!!!!!')
        attention_masks.append(batch['attention_mask'])
        input_ids_all.append(batch['input_ids'])
        indices.append(batch['indices'])
        attention_mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                ref_response = model(
                    input_ids,attention_mask=attention_mask,output_hidden_states=True
                )
                ref_prob = torch.softmax(ref_response.logits[:,:-1], dim=-1).to(
                    torch.float16
                )
                ref_prob_u = ref_prob[:, :, unq_tokens_ts]
                label_indices = input_ids[:,1:].unsqueeze(-1)
                one_hot = torch.zeros_like(ref_prob,dtype=torch.float16)
                one_hot.scatter_(dim=-1, index=label_indices, value=1.0)
                one_hot_u = one_hot[:, :, unq_tokens_ts]
                pbdiff = one_hot_u - ref_prob_u
                pb_diffs.append(pbdiff.cpu())
                embedding_list.append(ref_response.hidden_states[-1][:,:-1]
                                    .detach()
                                    .cpu()
                                    .to(torch.float16)
                                    )
    return input_ids_all, pb_diffs, embedding_list,attention_masks,indices
    
def batch_emb(a_emb,a_diffs,batch_size,axis = 1):
    length = a_emb.shape[axis]
    iters = np.ceil(length/batch_size)
    output = []
    for i in range(int(iters)):
        d1 = a_emb[:,batch_size*i:batch_size*(i+1),:]
        d2 = a_diffs[:,batch_size*i:batch_size*(i+1),:]
        c1 = d1[:,:,None,:] * d2[:,:,:,None]
        output.append(torch.sum(c1,dim=axis))
    all_emb = torch.stack(output,dim=axis)
    return all_emb.sum(axis)

def calculate_embsum(
    embedding_all,
    pb_diffs_all,
    batch_size,
    len_batch = 100,
    disable_diff = False,
    device = 'cuda:0',
    separate_token_pb_similarity = False,
):
    all_embdiffs = []
    token_emb_sums = []
    avg_pb_diffs = []
    for i in range(int(np.ceil(embedding_all.shape[0]/batch_size))):
        start_id = i*batch_size
        end_id = (i+1)*batch_size
        embedding_select = embedding_all[start_id:end_id].to(torch.bfloat16).to(device)
        if separate_token_pb_similarity:
            pb_diffs_select = pb_diffs_all[start_id:end_id].to(torch.bfloat16).to(device)
            valid_counts = (embedding_select.abs().sum(dim=-1) > 0).sum(dim=1, keepdim=True).clamp(min=1)
            valid_counts = valid_counts.to(pb_diffs_select.dtype).unsqueeze(-1)
            token_emb_sums.append(embedding_select.sum(1, keepdim=True).cpu())
            avg_pb_diffs.append((pb_diffs_select.sum(1, keepdim=True) ).cpu())
        elif disable_diff:
            all_emb_diff = embedding_select.sum(1,keepdim=True)
            all_embdiffs.append(all_emb_diff.cpu())
        else:
            pb_diffs_select = pb_diffs_all[start_id:end_id].to(torch.bfloat16).to(device)
            all_emb_diff = batch_emb(embedding_select,pb_diffs_select,len_batch,axis = 1)
            all_embdiffs.append(all_emb_diff.cpu())
    if separate_token_pb_similarity:
        return torch.cat(token_emb_sums, dim=0), torch.cat(avg_pb_diffs, dim=0)
    all_embdiffs = torch.cat(all_embdiffs,dim=0)
    return all_embdiffs

def calculate_forvalue(train_all_embdiffs,test_all_embdiffs,batch_size = 20):
    train_2_test_values = []
    for i in range(int(np.ceil(test_all_embdiffs.shape[0]/batch_size))):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size,test_all_embdiffs.shape[0])
        value_list = []
        for j in range(int(np.ceil(train_all_embdiffs.shape[0]/batch_size))):
            s_train = j*batch_size
            e_train = min((j+1)*batch_size,train_all_embdiffs.shape[0])
            
            values = (
                test_all_embdiffs[start_id:end_id,None] 
                * train_all_embdiffs[None,s_train:e_train]
            )

            value_list.append(values.sum(-1).sum(-1))

        values = torch.cat(value_list,dim=1)
        train_2_test_values.append(values)
    return torch.cat(train_2_test_values,dim=0)
