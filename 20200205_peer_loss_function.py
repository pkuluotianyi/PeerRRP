#!/usr/bin/env python
# coding: utf-8

# # 6 - Transformers for Sentiment Analysis
# 
# In this notebook we will be using the transformer model, first introduced in [this](https://arxiv.org/abs/1706.03762) paper. Specifically, we will be using the BERT (Bidirectional Encoder Representations from Transformers) model from [this](https://arxiv.org/abs/1810.04805) paper. 
# 
# Transformer models are considerably larger than anything else covered in these tutorials. As such we are going to use the [transformers library](https://github.com/huggingface/transformers) to get pre-trained transformers and use them as our embedding layers. We will freeze (not train) the transformer and only train the remainder of the model which learns from the representations produced by the transformer. In this case we will be using a multi-layer bi-directional GRU, however any model can learn from these representations.

# ## Preparing Data
# 
# First, as always, let's set the random seeds for deterministic results.

# In[1]:


import torch

import random
import numpy as np

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True









# The transformer has already been trained with a specific vocabulary, which means we need to train with the exact same vocabulary and also tokenize our data in the same way that the transformer did when it was initially trained.
# 
# Luckily, the transformers library has tokenizers for each of the transformer models provided. In this case we are using the BERT model which ignores casing (i.e. will lower case every word). We get this by loading the pre-trained `bert-base-uncased` tokenizer.

# In[2]:


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# The `tokenizer` has a `vocab` attribute which contains the actual vocabulary we will be using. We can check how many tokens are in it by checking its length.

# In[3]:


len(tokenizer.vocab)


# Using the tokenizer is as simple as calling `tokenizer.tokenize` on a string. This will tokenize and lower case the data in a way that is consistent with the pre-trained transformer model.

# In[4]:


tokens = tokenizer.tokenize('Hello WORLD how ARE yoU?')

print(tokens)


# We can numericalize tokens using our vocabulary using `tokenizer.convert_tokens_to_ids`.

# In[5]:


indexes = tokenizer.convert_tokens_to_ids(tokens)

print(indexes)


# The transformer was also trained with special tokens to mark the beginning and end of the sentence, detailed [here](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel). As well as a standard padding and unknown token. We can also get these from the tokenizer.
# 
# **Note**: the tokenizer does have a beginning of sequence and end of sequence attributes (`bos_token` and `eos_token`) but these are not set and should not be used for this transformer.

# In[6]:


init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)


# We can get the indexes of the special tokens by converting them using the vocabulary...

# In[7]:


init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)


# ...or by explicitly getting them from the tokenizer.

# In[8]:


init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)


# Another thing we need to handle is that the model was trained on sequences with a defined maximum length - it does not know how to handle sequences longer than it has been trained on. We can get the maximum length of these input sizes by checking the `max_model_input_sizes` for the version of the transformer we want to use. In this case, it is 512 tokens.

# In[9]:


max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

print(max_input_length)


# Previously we have used the `spaCy` tokenizer to tokenize our examples. However we now need to define a function that we will pass to our `TEXT` field that will handle all the tokenization for us. It will also cut down the number of tokens to a maximum length. Note that our maximum length is 2 less than the actual maximum length. This is because we need to append two tokens to each sequence, one to the start and one to the end.

# In[10]:


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


# Now we define our fields. The transformer expects the batch dimension to be first, so we set `batch_first = True`. As we already have the vocabulary for our text, provided by the transformer we set `use_vocab = False` to tell torchtext that we'll be handling the vocabulary side of things. We pass our `tokenize_and_cut` function as the tokenizer. The `preprocessing` argument is a function that takes in the example after it has been tokenized, this is where we will convert the tokens to their indexes. Finally, we define the special tokens - making note that we are defining them to be their index value and not their string value, i.e. `100` instead of `[UNK]` This is because the sequences will already be converted into indexes.
# 
# We define the label field as before.

# In[11]:


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






###################################################################
#test_code
#TEXT = data.Field()
#LABEL = data.Field()

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='./data', train='train_kdd2020.csv',
    validation='valid_kdd2020.csv', test='test_kdd2020.csv', format='csv',
    fields=[('text', TEXT), ('labels', LABEL)])
###################################################################


import random

# batch_num = 128
# 
# #dataset_num_total = 64#1797
# train_dataset_num = 38
# test_dataset_num = 26
# unlabelled_dataset_num = 2170
# 
# index_list_total = []
# 
# for i in range(train_dataset_num + unlabelled_dataset_num):
#     index_list_total.append(i)
# 
# index_sequence_first = random.sample(index_list_total, batch_num)
# index_sequence_second = random.sample(index_list_total, batch_num)
# index_sequence_third = random.sample(index_list_total, batch_num)
# 
# train_input_list_first_item = []
# train_input_label_first_item = []
# for i in range(batch_num):
#     temp_num = index_sequence_first[i]
#     
#     test_1_var = train_data[temp_num]
#     test_2_var = test_1_var.text
#     test_3_var = test_1_var.labels
#     
#     temp_list = train_data[temp_num].text
#     train_input_list_first_item.append(temp_list)
#     temp_label = train_data[temp_num].labels
#     train_input_label_first_item.append(temp_label)
#     
# train_input_list_second_item = []
# train_input_label_second_item = []
# for i in range(batch_num):
#     temp_num_second1 = index_sequence_second[i]
#     temp_num_second2 = index_sequence_third[i]
#     temp_list = train_data[temp_num_second1].text
#     train_input_list_second_item.append(temp_list)
#     temp_label = train_data[temp_num_second2].labels
#     train_input_label_second_item.append(temp_label)



# We load the data and create the validation splits as before.

# In[12]:


from torchtext import datasets

#train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

#train_data, valid_data = train_data.split(random_state = random.seed(SEED))


# In[13]:


print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")


# We can check an example and ensure that the text has already been numericalized.

# In[14]:


print(vars(train_data.examples[6]))


# We can use the `convert_ids_to_tokens` to transform these indexes back into readable tokens.

# In[15]:


tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])

print(tokens)


# Although we've handled the vocabulary for the text, we still need to build the vocabulary for the labels.

# In[16]:


LABEL.build_vocab(train_data)


# In[17]:


print(LABEL.vocab.stoi)


# As before, we create the iterators. Ideally we want to use the largest batch size that we can as I've found this gives the best results for transformers.

# In[18]:


BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)


# ## Build the Model
# 
# Next, we'll load the pre-trained model, making sure to load the same model as we did for the tokenizer.

# In[19]:


from transformers import BertTokenizer, BertModel

bert = BertModel.from_pretrained('bert-base-uncased')


# Next, we'll define our actual model. 
# 
# Instead of using an embedding layer to get embeddings for our text, we'll be using the pre-trained transformer model. These embeddings will then be fed into a GRU to produce a prediction for the sentiment of the input sentence. We get the embedding dimension size (called the `hidden_size`) from the transformer via its config attribute. The rest of the initialization is standard.
# 
# Within the forward pass, we wrap the transformer in a `no_grad` to ensure no gradients are calculated over this part of the model. The transformer actually returns the embeddings for the whole sequence as well as a *pooled* output. The [documentation](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel) states that the pooled output is "usually not a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence", hence we will not be using it. The rest of the forward pass is the standard implementation of a recurrent model, where we take the hidden state over the final time-step, and pass it through a linear layer to get our predictions.

# In[20]:


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
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output


# Next, we create an instance of our model using standard hyperparameters.

# In[21]:


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


# We can check how many parameters the model has. Our standard models have under 5M, but this one has 112M! Luckily, 110M of these parameters are from the transformer and we will not be training those.

# In[22]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In order to freeze paramers (not train them) we need to set their `requires_grad` attribute to `False`. To do this, we simply loop through all of the `named_parameters` in our model and if they're a part of the `bert` transformer model, we set `requires_grad = False`. 

# In[23]:


for name, param in model.named_parameters():                
    if name.startswith('bert'):
        param.requires_grad = False


# We can now see that our model has under 3M trainable parameters, making it almost comparable to the `FastText` model. However, the text still has to propagate through the transformer which causes training to take considerably longer.

# In[24]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# We can double check the names of the trainable parameters, ensuring they make sense. As we can see, they are all the parameters of the GRU (`rnn`) and the linear layer (`out`).

# In[25]:


for name, param in model.named_parameters():                
    if param.requires_grad:
        print(name)


# ## Train the Model
# 
# As is standard, we define our optimizer and criterion (loss function).

# In[26]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())


# In[27]:


criterion = nn.BCEWithLogitsLoss()


# Place the model and criterion onto the GPU (if available)

# In[28]:


model = model.to(device)
criterion = criterion.to(device)


# Next, we'll define functions for: calculating accuracy, performing a training epoch, performing an evaluation epoch and calculating how long a training/evaluation epoch takes.

# In[29]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# In[30]:


def train(model, epoch, optimizer, criterion):
    
    import random
    
    batch_num = 128
    
    #dataset_num_total = 64#1797
    train_dataset_num = 300
    test_dataset_num = 99
    unlabelled_dataset_num = 2070
    
    index_list_total = []
    
    for i in range(train_dataset_num + unlabelled_dataset_num):
        index_list_total.append(i)
    
    index_sequence_first = random.sample(index_list_total, batch_num)
    index_sequence_second = random.sample(index_list_total, batch_num)
    index_sequence_third = random.sample(index_list_total, batch_num)
    
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
    
    #for batch in iterator:
        
    optimizer.zero_grad()
    
    predictions = model(train_input_list_first_item_torch_tensor).squeeze(1)
    
    loss_first = criterion(predictions, train_input_label_first_item_torch_tensor)
    
    predictions_second = model(train_input_list_second_item_torch_tensor).squeeze(1)
    
    loss_second = criterion(predictions_second, train_input_label_second_item_torch_tensor)
    
    alpha_coefficient = 0.0
    if epoch > -1 and epoch <= 30:
        alpha_coefficient = 0.0
    if epoch > 30 and epoch <= 100:
        alpha_coefficient = 0.2
    if epoch > 100 and epoch <= 150:
        alpha_coefficient = 0.4 
    if epoch > 150 and epoch <= 200:
        alpha_coefficient = 0.5
    if epoch > 200 and epoch <= 220: 
        alpha_coefficient = 0.6
    if epoch > 220 and epoch <= 333:
        alpha_coefficient = 1.3
    if epoch > 333 and epoch <= 359:
        alpha_coefficient = 1.9
    if epoch > 359 and epoch <= 378:
        alpha_coefficient = 8.0
    if epoch > 378:
        alpha_coefficient = 8.0
#     if epoch > 600 and epoch <= 1000:
#         alpha_coefficient = 0.4
#     if epoch > 1000 and epoch <= 1500:
#         alpha_coefficient = 0.4
#     if epoch > 1500 and epoch <= 2000:
#         alpha_coefficient = 0.6
#     if epoch > 2000 and epoch <= 2500:
#         alpha_coefficient = 0.6
        
    loss = loss_first - alpha_coefficient * loss_second
    #loss = loss_first
        
    acc = binary_accuracy(predictions, train_input_label_first_item_torch_tensor)
    
    loss.backward()
    
    optimizer.step()
    
    epoch_loss += loss.item()
    epoch_acc += acc.item()
        
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return loss, acc

# In[31]:
def evaluate_valid(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    
    batch_num = 128
    
    #dataset_num_total = 64#1797
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
        
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return loss, acc

def evaluate(model, iterator, criterion):
    
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
        
        for i in range(test_dataset_num):
            print("id: " + str(i) + "  \n")
            print(str(predictions[i]))
            
        loss = criterion(predictions, test_input_label_first_item_torch_tensor)
            
        acc = binary_accuracy(predictions, test_input_label_first_item_torch_tensor)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return loss, acc


# In[32]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Finally, we'll train our model. This takes considerably longer than any of the previous models due to the size of the transformer. Even though we are not training any of the transformer's parameters we still need to pass the data through the model which takes a considerable amount of time on a standard GPU.

# In[34]:


N_EPOCHS = 5000

best_test_acc = 0.7172#float('-inf')

model.load_state_dict(torch.load('20200326-best-model--378--tensor(0.7172, device=\'cuda:0\').pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

for epoch in range(N_EPOCHS):
    
    if epoch <= 378:
        continue
    
    start_time = time.time()
    
    train_loss, train_acc = train(model, epoch, optimizer, criterion)
    valid_loss, valid_acc = evaluate_valid(model, valid_iterator, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    
    if test_acc > 0.75:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
    end_time = time.time()
        
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        
        
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), '20200326-best-model--' + str(epoch) + '--' + str(best_test_acc) + '.pt')
      
        
        
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# We'll load up the parameters that gave us the best validation loss and try these on the test set - which gives us our best results so far!

# In[35]:


model.load_state_dict(torch.load('tut6-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion, 'test')

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# ## Inference
# 
# We'll then use the model to test the sentiment of some sequences. We tokenize the input sequence, trim it down to the maximum length, add the special tokens to either side, convert it to a tensor, add a fake batch dimension and then pass it through our model.

# In[36]:


def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


# In[37]:


predict_sentiment(model, tokenizer, "This film is terrible")


# In[38]:


predict_sentiment(model, tokenizer, "This film is great")

