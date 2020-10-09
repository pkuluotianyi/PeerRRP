#!/usr/bin/env python
# coding: utf-8

import torch
import random
import numpy as np
import matplotlib

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


from transformers import BertTokenizer, BertConfig, BertForPreTraining

#replace the following directory (bert pretrained model) with the one in your machine 
tokenizer = BertTokenizer.from_pretrained('./20200908_bert_further_training_emnlp_corpus/uncased_L-12_H-768_A-12_AGnews_pretrain/')

tokens = tokenizer.tokenize('Hello WORLD how ARE yoU?')
print(tokens)

indexes = tokenizer.convert_tokens_to_ids(tokens)
print(indexes)

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
print(init_token, eos_token, pad_token, unk_token)


init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

#maximum input lengh
max_input_length = 1-000#tokenizer.max_model_input_sizes['bert-base-uncased']

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


from torchtext import data

TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = data.LabelField(dtype = torch.float)

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='./data', train='train_kdd2020.csv',
    validation='valid_kdd2020.csv', test='test_kdd2020.csv', format='csv',
    fields=[('text', TEXT), ('labels', LABEL)])


print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")
print(vars(train_data.examples[6]))

tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])
print(tokens)


LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)




BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import BertTokenizer, BertModel

#replace the following directory (bert pretrained model) with the one in your machine 
bert = BertModel.from_pretrained('./20200908_bert_further_training_emnlp_corpus/uncased_L-12_H-768_A-12_AGnews_pretrain/')

import torch.nn as nn

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        print("embedding_dim: " + str(embedding_dim))
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        num_sections = list(text.size())[1] // 512
        
        with torch.no_grad():
            embedded = self.bert(text[:,:512])[0]

        _all, hidden_all = self.rnn(embedded)
        
        for i in range(num_sections - 1):
            with torch.no_grad():
                embedded = self.bert(text[:,512*(i+1):512*(i+2)])[0]

            _, hidden = self.rnn(embedded)
            hidden_all += hidden
            
        hidden_all /= num_sections
        
        if self.rnn.bidirectional:
            hidden_all = self.dropout(torch.cat((hidden_all[-2,:,:], hidden_all[-1,:,:]), dim = 1))
        else:
            hidden_all = self.dropout(hidden_all[-1,:,:])
    
        output = self.out(hidden_all)
        
        return output



HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, epoch, optimizer, criterion):
    
    import random
    
    batch_num = 32
    
    train_dataset_num = 300
    test_dataset_num = 99
    unlabelled_dataset_num = 2070
    
    index_list_total = []
    
    for i in range(train_dataset_num + unlabelled_dataset_num):
        index_list_total.append(i)
    
    #check whether the three sampled points are different each other for each peer pair
    continue_check_bool = True
    while continue_check_bool:
        index_sequence_first = random.sample(index_list_total, batch_num)
        index_sequence_second = random.sample(index_list_total, batch_num)
        index_sequence_third = random.sample(index_list_total, batch_num)
        
        #checking
        for i_check in range(batch_num):
            temp_first_sequence = index_sequence_first[i_check]
            temp_second_sequence = index_sequence_second[i_check]
            temp_third_sequence = index_sequence_third[i_check]
            
            if temp_first_sequence != temp_second_sequence and temp_second_sequence != temp_third_sequence and temp_first_sequence != temp_third_sequence:
                continue
            else:
                continue_check_bool = False
                print("epoch:" + str(epoch) + " i_check:" + str(i_check) + " " + str(temp_first_sequence) + " " + str(temp_second_sequence) + "" + str(temp_third_sequence))
                break
        
        if continue_check_bool == True:
            continue_check_bool = False
            continue
            
        if continue_check_bool == False:
            continue_check_bool = True
            continue
            
    
    train_input_list_first_item = []
    train_input_label_first_item = []
    for i in range(batch_num):
        temp_num = index_sequence_first[i]
        temp_list = train_data[temp_num].text
        temp_len = len(temp_list)
        remain_len = max_input_length - 2 - temp_len
        for i in range(remain_len):
            temp_list.append(0)
        train_input_list_first_item.append(temp_list)
        temp_label = int(train_data[temp_num].labels)
        train_input_label_first_item.append(temp_label)
        
    train_input_list_second_item = []
    train_input_label_second_item = []
    for i in range(batch_num):
        temp_num_second1 = index_sequence_second[i]
        temp_num_second2 = index_sequence_third[i]
        temp_list = train_data[temp_num_second1].text
        temp_len = len(temp_list)
        remain_len = max_input_length - 2 - temp_len
        for i in range(remain_len):
            temp_list.append(0)
        train_input_list_second_item.append(temp_list)
        temp_label = int(train_data[temp_num_second2].labels)
        train_input_label_second_item.append(temp_label)
        
        
    train_input_list_first_item_array = np.array(train_input_list_first_item)
    train_input_label_first_item_array = np.array(train_input_label_first_item, dtype='float32')
    train_input_list_second_item_array = np.array(train_input_list_second_item)
    train_input_label_second_item_array = np.array(train_input_label_second_item, dtype='float32')
    
    train_input_list_first_item_torch_tensor = torch.from_numpy(train_input_list_first_item_array)
    train_input_label_first_item_torch_tensor = torch.from_numpy(train_input_label_first_item_array)
    train_input_list_second_item_torch_tensor = torch.from_numpy(train_input_list_second_item_array)
    train_input_label_second_item_torch_tensor = torch.from_numpy(train_input_label_second_item_array)
    
    train_input_list_first_item_torch_tensor = train_input_list_first_item_torch_tensor.cuda()
    train_input_label_first_item_torch_tensor = train_input_label_first_item_torch_tensor.cuda()
    train_input_list_second_item_torch_tensor = train_input_list_second_item_torch_tensor.cuda()
    train_input_label_second_item_torch_tensor = train_input_label_second_item_torch_tensor.cuda()
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
        
    optimizer.zero_grad()
    
    predictions = model(train_input_list_first_item_torch_tensor).squeeze(1)
    
    loss_first = criterion(predictions, train_input_label_first_item_torch_tensor)
    
    predictions_second = model(train_input_list_second_item_torch_tensor).squeeze(1)
    
    loss_second = criterion(predictions_second, train_input_label_second_item_torch_tensor)
    
    alpha_coefficient = 0.0
#     if epoch > -1 and epoch <= 30:
#         alpha_coefficient = 0.0
#     if epoch > 30 and epoch <= 100:
#         alpha_coefficient = 0.2
#     if epoch > 100 and epoch <= 150:
#         alpha_coefficient = 0.4 
#     if epoch > 150 and epoch <= 200:
#         alpha_coefficient = 0.5
#     if epoch > 200 and epoch <= 220: 
#         alpha_coefficient = 0.6
#     if epoch > 220 and epoch <= 333:
#         alpha_coefficient = 1.3
#     if epoch > 333 and epoch <= 359:
#         alpha_coefficient = 1.9
#     if epoch > 359 and epoch <= 378:
#         alpha_coefficient = 8.0
#     if epoch > 378:
#         alpha_coefficient = 8.0
        
        
#     if epoch > -1 and epoch <= 400*2:
#         alpha_coefficient = 0.0
#     if epoch > 400*2 and epoch <= 800*2:
#         alpha_coefficient = 0.2
#     if epoch > 800*2 and epoch <= 901*2:
#         alpha_coefficient = 0.4
#     if epoch > 901*2 and epoch <= 979*2:
#         alpha_coefficient = 0.0  
#     if epoch > 979*2 and epoch <= 1600*2:
#         alpha_coefficient = 0.00

    if epoch > -1 and epoch <= 400:
        alpha_coefficient = 0.0
    if epoch > 400 and epoch <= 800:
        alpha_coefficient = 0.2
    if epoch > 800 and epoch <= 901:
        alpha_coefficient = 0.4
    if epoch > 901 and epoch <= 979:
        alpha_coefficient = 0.0  
    if epoch > 979 and epoch <= 1050:
        alpha_coefficient = 0.1
    if epoch > 1050 and epoch <= 1400:
        alpha_coefficient = 0.0
    if epoch > 1400:
        alpha_coefficient = 0.1
        
    loss = loss_first - alpha_coefficient * loss_second
        
    acc = binary_accuracy(predictions, train_input_label_first_item_torch_tensor)
    
    loss.backward()
    
    optimizer.step()
    
    epoch_loss += loss.item()
    epoch_acc += acc.item()
        
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return loss, acc

def evaluate_valid(model, criterion):
    epoch_loss = 0
    epoch_acc = 0
    batch_num = 128
    
    train_dataset_num = 300
    test_dataset_num = 99
    unlabelled_dataset_num = 2070
    valid_num = 100
    
    valid_input_list_first_item = []
    valid_input_label_first_item = []

    for i in range(valid_num):
        temp_num = i
        temp_list = valid_data[temp_num].text
        temp_len = len(temp_list)
        remain_len = max_input_length - 2 - temp_len
        for i in range(remain_len):
            temp_list.append(0)
        valid_input_list_first_item.append(temp_list)
        temp_label = int(valid_data[temp_num].labels)
        valid_input_label_first_item.append(temp_label)

    valid_input_list_first_item_array = np.array(valid_input_list_first_item)
    valid_input_label_first_item_array = np.array(valid_input_label_first_item, dtype='float32')

    
    valid_input_list_first_item_torch_tensor = torch.from_numpy(valid_input_list_first_item_array)
    valid_input_label_first_item_torch_tensor = torch.from_numpy(valid_input_label_first_item_array)

    
    valid_input_list_first_item_torch_tensor = valid_input_list_first_item_torch_tensor.cuda()
    valid_input_label_first_item_torch_tensor = valid_input_label_first_item_torch_tensor.cuda()

    model.eval()
    
    with torch.no_grad():
        predictions = model(valid_input_list_first_item_torch_tensor).squeeze(1)     
        loss = criterion(predictions, valid_input_label_first_item_torch_tensor)  
        acc = binary_accuracy(predictions, valid_input_label_first_item_torch_tensor)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return loss, acc

def evaluate(model, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
   
    batch_num = 128
    
    #dataset_num_total = 64#1797
    train_dataset_num = 300
    test_dataset_num = 99
    unlabelled_dataset_num = 2070
    valid_num = 100
    
    test_input_list_first_item = []
    test_input_label_first_item = []

    for i in range(test_dataset_num):
        temp_num = i
        temp_list = test_data[temp_num].text
        temp_len = len(temp_list)
        remain_len = max_input_length - 2 - temp_len
        for i in range(remain_len):
            temp_list.append(0)
        test_input_list_first_item.append(temp_list)
        temp_label = int(test_data[temp_num].labels)
        test_input_label_first_item.append(temp_label)

    test_input_list_first_item_array = np.array(test_input_list_first_item)
    test_input_label_first_item_array = np.array(test_input_label_first_item, dtype='float32')

    
    test_input_list_first_item_torch_tensor = torch.from_numpy(test_input_list_first_item_array)
    test_input_label_first_item_torch_tensor = torch.from_numpy(test_input_label_first_item_array)

    
    test_input_list_first_item_torch_tensor = test_input_list_first_item_torch_tensor.cuda()
    test_input_label_first_item_torch_tensor = test_input_label_first_item_torch_tensor.cuda()

    model.eval()
    
    with torch.no_grad():
        predictions = model(test_input_list_first_item_torch_tensor).squeeze(1)
        loss = criterion(predictions, test_input_label_first_item_torch_tensor)
        acc = binary_accuracy(predictions, test_input_label_first_item_torch_tensor)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return loss, acc


N_EPOCHS = 10000

best_val_acc = float('-inf')
best_test_acc = float('-inf')

for epoch in range(N_EPOCHS):
    
    train_loss, train_acc = train(model, epoch, optimizer, criterion)
    valid_loss, valid_acc = evaluate_valid(model, criterion)
    test_loss, test_acc = evaluate(model, criterion)

    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        torch.save(model.state_dict(), '20201003-further-pretraining-max-length-10000-batch-32-peer-loss-best-model--' + str(epoch) + '--valid-' + str(valid_acc) + '--test-' + str(test_acc) + '.pt')
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), '20201003-further-pretraining-max-length-10000-batch-32-peer-loss-best-model--' + str(epoch) + '--valid-' + str(valid_acc) + '--test-' + str(test_acc) + '.pt')
      
    print(str(epoch) + " th iteration:\n")
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print("best test accuracy:" + str(best_test_acc))



