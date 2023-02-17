#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI, Request, Response, Body
import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification, BertTokenizer
import re


# In[2]:


MAX_LEN = 400
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

##########################################################################
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
   
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# In[12]:


class BertClassifier(nn.Module):
    
    def __init__(self, freeze_bert = False):
        
        super(BertClassifier, self).__init__()
        D_in = 768
        H = 100
        D_out = 2
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.classifier = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out))
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_inds, attention_mask):
        
        outputs = self.bert(input_inds, attention_mask)
        
        last_hidden = outputs[0][:, 0, :]
        
        logits = self.classifier(last_hidden)
        
        return logits
        


# In[13]:


model = BertClassifier(freeze_bert=False)


# In[14]:


app = FastAPI()
# Load your pre-trained model
#model = (freeze_bert=False)
#state_dict = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(torch.load('checkpoint-BERT-nofreeze(1).pth', map_location=torch.device('cpu')))
#model.load_state_dict(state_dict)
model.eval()


# In[20]:



@app.post("/predict")
async def predict(x: str):
    # Preprocess the input text
    test_inputs, test_masks  = preprocessing_for_bert([x])
    

    with torch.no_grad():
        # Convert the preprocessed text to a tensor
        outs = model(test_inputs, test_masks)
        prediction = torch.argmax(outs, dim=1).flatten()


    # Return the prediction in the response
    if prediction == 0: 
       return str('Profanity')
    else: return str('Clean')


