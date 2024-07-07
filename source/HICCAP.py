from torch import nn
from FeatureEncoding import FeatureEncoding
from HCA import HCA
import ComicMischiefTasks as Tasks
import torch
import config as C
import os

class HICCAP(nn.Module):
    def __init__(self, heads=None, encoding=None, hca=None):
        super(HICCAP, self).__init__()
        if heads == None:
            raise ValueError("Heads should be either {}".format(C.supported_heads))     
        for head in heads:
            if head not in C.supported_heads:
                raise ValueError("Heads should be either {}".format(C.supported_heads))
        device = C.device
        if encoding == None:
            encoding = FeatureEncoding()
        if hca == None:
            hca = HCA()
        self.feature_encoding = encoding
        self.hca = hca

        self.feature_encoding.to(device)
        self.hca.to(device)
        self.task_heads = {}
        for head in heads:
            if head == "binary":
                self.task_heads[head] = Tasks.ComicMischiefBinary()
            elif head == "mature":
                self.task_heads[head] = Tasks.ComicMischiefMature()
            elif head == "gory":
                self.task_heads[head] = Tasks.ComicMischiefGory()
            elif head == "slapstick":
                self.task_heads[head] = Tasks.ComicMischiefSlapstick()
            else:
                assert(head == "sarcasm")
                self.task_heads[head] = Tasks.ComicMischiefSarcasm()
            self.task_heads[head].to(device)

    def set_training_mode(self):
        self.feature_encoding.train()
        self.hca.train()
        for head in self.task_heads:
            self.task_heads[head].train()
        
    def save(self):
        if os.path.exists("modular_fe.pth"):
            os.remove("modular_fe.pth")
        torch.save(self.feature_encoding.state_dict(), "/tmp/modular_fe.pth")
        
        if os.path.exists("modular_hca.pth"):
            os.remove("modular_hca.pth")
        torch.save(self.hca.state_dict(), "/tmp/modular_hca.pth")
    
    def load(self, fe_file, hca_file):
        device = C.device
        modular_fe = torch.load(fe_file, map_location=device)
        self.feature_encoding.load_state_dict(modular_fe['model_state'], strict=False)
        modular_hca = torch.load(hca_file, map_location=device)
        self.hca.load_state_dict(modular_hca['model_state'], strict=False)

    def set_eval_mode(self):
        self.feature_encoding.eval()
        self.hca.eval()
        for head in self.task_heads:
            self.task_heads[head].eval()

    def get_model_params(self):
        params = [{'params': self.feature_encoding.parameters()},
                 {'params': self.hca.parameters()}]
        for head in self.task_heads:
            params.append({'params': self.task_heads[head].parameters()})
        return params

    def forward(self, sentences, mask, image, image_mask, audio, audio_mask):
        hidden, rnn_img_encoded, extended_image_attention_mask,\
            rnn_audio_encoded, extended_audio_attention_mask,\
                extended_attention_mask = self.feature_encoding(sentences, 
                                                                mask, 
                                                                image, 
                                                                image_mask, 
                                                                audio, 
                                                                audio_mask)
        
        output_text, output_audio, output_image = self.hca(hidden,
                                                           rnn_img_encoded,
                                                           extended_image_attention_mask, 
                                                           rnn_audio_encoded, 
                                                           extended_audio_attention_mask, 
                                                           extended_attention_mask)
        
        return output_text, output_audio, output_image
    
    def forward_pass(self, sentences, mask, image, image_mask, audio, audio_mask, actual):
        output_text, output_audio, output_image = self.forward(sentences, mask, image, 
                                                               image_mask, audio, audio_mask)
        outputs = {}
        for head in self.task_heads:
            output = self.task_heads[head].forward_pass(output_text, 
                                                        output_audio, 
                                                        output_image, 
                                                        actual[head])
            outputs.update(output)
        return outputs
    
    def eval_pass(self, sentences, mask, image, image_mask, audio, audio_mask):
        output_text, output_audio, output_image = self.forward(sentences, mask, image, 
                                                               image_mask, 
                                                               audio, 
                                                               audio_mask)
        outputs = {}
        for head in self.task_heads:
            output = self.task_heads[head].eval_pass(output_text, output_audio, output_image)
            outputs.update(output)
        return outputs