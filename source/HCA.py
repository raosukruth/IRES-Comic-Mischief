import torch
from torch import nn
import math
from transformers import BertTokenizer, BertModel
import numpy as np


class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()

        self.u = nn.Linear(dimension, dimension)
        self.v = nn.Parameter(torch.rand(dimension), requires_grad=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-10

    def forward(self, h, mask):
       u_it = self.u(h)
       u_it = self.tanh(u_it)
       
       alpha = torch.exp(torch.matmul(u_it, self.v))
       alpha = mask * alpha + self.epsilon
       denominator_sum = torch.sum(alpha, dim=-1, keepdim=True)
       alpha = mask * (alpha / denominator_sum)
       output = h * alpha.unsqueeze(2)
       output = torch.sum(output, dim=1)
       return output, alpha

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

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class HCA(nn.Module):
    def __init__(self):
        super(HCA, self).__init__()
        self.rnn_units = 512
        self.embedding_dim = 768
        dropout = 0.2

        self.att1 = BertOutAttention(self.embedding_dim)
        self.att1_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att1_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att2 = BertOutAttention(self.embedding_dim)
        self.att2_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att2_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        
        self.att3 = BertOutAttention(self.embedding_dim)
        self.att3_drop_norm1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.att3_drop_norm2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units*2  , self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),
        )
        self.sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units*2  , self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),
        )

        self.rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=2, bidirectional=True, batch_first = True)
        self.rnn_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 
        self.rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=2, bidirectional=True, batch_first = True)
        self.rnn_img_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 

        self.attention_audio = Attention(768)
        self.attention_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.attention_image = Attention(768)
        self.attention_image_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        ) 
        self.attention = Attention(768)
        self.attention_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.embedding_dim, eps=1e-5),
        )
        
    def forward(self, hidden, rnn_img_encoded, extended_image_attention_mask, 
                rnn_audio_encoded, extended_audio_attention_mask, extended_attention_mask):
        output_text = self.att1(hidden[-1], self.sequential_image(rnn_img_encoded), extended_image_attention_mask)
        output_text = self.att1_drop_norm1(output_text)
        output_text = self.att1(output_text, self.sequential_audio(rnn_audio_encoded), extended_audio_attention_mask)
        output_text = self.att1_drop_norm2(output_text)
        
        output_text = output_text + hidden[-1]

        output_audio = self.att2(self.sequential_audio(rnn_audio_encoded), self.sequential_image(rnn_img_encoded) ,extended_image_attention_mask)
        output_audio = self.att2_drop_norm1(output_audio)
        output_audio = self.att2(output_audio, hidden[-1], extended_attention_mask)   
        output_audio = self.att2_drop_norm2(output_audio)
        
        output_audio = output_audio + self.sequential_audio(rnn_audio_encoded)

        output_image = self.att3(self.sequential_image(rnn_img_encoded), hidden[-1], extended_attention_mask)
        output_image = self.att3_drop_norm1(output_image)
        output_image = self.att3(output_image, self.sequential_audio(rnn_audio_encoded) ,extended_audio_attention_mask)
        output_image = self.att3_drop_norm2(output_image)
        
        output_image = output_image + self.sequential_image(rnn_img_encoded)

        mask = torch.tensor(np.array([1]*output_text.size()[1])).to(next(self.parameters()).device) # cuda()
        audio_mask = torch.tensor(np.array([1]*output_audio.size()[1])).to(next(self.parameters()).device) # cuda()
        image_mask = torch.tensor(np.array([1]*output_image.size()[1])).to(next(self.parameters()).device) # cuda()

        #print("TEXT BEFORE SELF ATTENTION:", output_text.shape)
        output_text, attention_weights = self.attention(output_text, mask.float())
        output_text = self.attention_drop_norm(output_text)
        #print("TEXT AFTER SELF ATTENTION:", output_text.shape)
        output_audio, attention_weights = self.attention_audio(output_audio, audio_mask.float())
        output_audio = self.attention_audio_drop_norm(output_audio)
        #print("IMAGE BEFORE SELF ATTENTION", output_image.shape)
        output_image, attention_weights = self.attention_image(output_image, image_mask.float())
        output_image = self.attention_image_drop_norm(output_image)
        #print("IMAGE AFTER SELF ATTENTION", output_image.shape)

        return output_text, output_audio, output_image