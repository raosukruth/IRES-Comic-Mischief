import sys


sys.path.append('../../')
import torch
from torch import nn
from torch.nn import functional as F
from source.models.attention import *
from source import config as C
import pprint
import numpy as np
from transformers import BertTokenizer, BertModel
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

pp = pprint.PrettyPrinter(indent=4).pprint

debug = False
bidirectional = True

class BertOutAttention(nn.Module):
    def __init__(self, size, ctx_dim=None):
        super().__init__()
        if size % 12 != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (size, 12))
        self.num_attention_heads = 12
        self.attention_head_size = int(size / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =size
        self.query = nn.Linear(size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class Bert_Model(nn.Module):

    def __init__(self):
        super(Bert_Model, self).__init__()

        self.rnn_units = 512
        self.embedding_dim = 768
        
        dropout = 0.2
        att1 = BertOutAttention(self.embedding_dim)
        att1_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        att1_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        att2 = BertOutAttention(self.embedding_dim)
        att2_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        att2_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        
        att3 = BertOutAttention(self.embedding_dim)
        att3_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        att3_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 

        sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units*2  , self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),

        )
        sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units*2  , self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),

        )
        bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                                output_attentions=True)

        rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=2, bidirectional=True, batch_first = True)
        rnn_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 
        rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=2, bidirectional=True, batch_first = True)
        rnn_img_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 

        attention_audio = Attention(768)
        attention_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        attention_image = Attention(768)
        attention_image_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        attention = Attention(768)
        attention_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        )
        
        self.features = nn.ModuleList([bert, rnn_img, rnn_img_drop_norm, rnn_audio, rnn_audio_drop_norm, sequential_audio, sequential_image, att1, att1_drop_norm1, att1_drop_norm2, att2, att2_drop_norm1, att2_drop_norm2, att3, att3_drop_norm1, att3_drop_norm2, attention, attention_drop_norm, attention_audio, attention_audio_drop_norm, attention_image, attention_image_drop_norm])
       
        self.img_audio_text_linear_mature = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

 
        self.img_audio_text_linear_gory = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

        self.img_audio_text_linear_slapstick = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

        self.img_audio_text_linear_sarcasm = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )


    def forward(self, sentences,mask,image, image_mask, audio, audio_mask):
        
        
        hidden, _ = self.features[0](sentences)[-2:]
        
        rnn_img_encoded, (hid, ct) = self.features[1](image)
        rnn_img_encoded = self.features[2](rnn_img_encoded)
        rnn_audio_encoded, (hid_audio, ct_audio) = self.features[3](audio)
        rnn_audio_encoded = self.features[4](rnn_audio_encoded)
        
        extended_attention_mask= mask.float().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
      
        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
 
        output_text = self.features[7](hidden[-1],   self.features[6](rnn_img_encoded) ,extended_image_attention_mask)
        output_text = self.features[8](output_text)
        output_text = self.features[7](output_text, self.features[5](rnn_audio_encoded) ,extended_audio_attention_mask)
        output_text = self.features[9](output_text)
        
        output_text = output_text + hidden[-1]

        output_audio = self.features[10](self.features[5](rnn_audio_encoded), self.features[6](rnn_img_encoded) ,extended_image_attention_mask )
        output_audio = self.features[11](output_audio)
        output_audio = self.features[10](output_audio, hidden[-1], extended_attention_mask)   
        output_audio = self.features[12](output_audio)
        
        output_audio = output_audio + self.features[5](rnn_audio_encoded)

        output_image = self.features[13](self.features[6](rnn_img_encoded),self.features[5](rnn_audio_encoded) ,extended_audio_attention_mask )
        output_image = self.features[14](output_image)
        output_image = self.features[13](output_image, hidden[-1], extended_attention_mask )
        output_image = self.features[15](output_image)
        
        output_image = output_image + self.features[6](rnn_img_encoded)

  
  
        mask = torch.tensor(np.array([1]*output_text.size()[1])).cuda()
        audio_mask = torch.tensor(np.array([1]*output_audio.size()[1])).cuda()
        image_mask = torch.tensor(np.array([1]*output_image.size()[1])).cuda()

        output_text, attention_weights = self.features[16](output_text, mask.float())
        output_text = self.features[17](output_text)
        output_audio,  attention_weights = self.features[18](output_audio, audio_mask.float())
        output_audio = self.features[19](output_audio)
        output_image,  attention_weights = self.features[20](output_image, image_mask.float())
        output_image = self.features[21](output_image)

        audio_text_image  = torch.cat([output_text,output_audio,output_image], dim=-1)

        output1 = F.softmax(self.img_audio_text_linear_mature(audio_text_image), -1)
        output2 = F.softmax(self.img_audio_text_linear_gory(audio_text_image), -1)
        output3 = F.softmax(self.img_audio_text_linear_slapstick(audio_text_image), -1)
        output4 = F.softmax(self.img_audio_text_linear_sarcasm(audio_text_image), -1)
        
        sequential_output = []   
        return [output1,output2,output3,output4], sequential_output
