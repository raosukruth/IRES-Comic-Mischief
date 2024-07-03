from torch import nn
from FeatureEncoding import FeatureEncoding
from HCA import HCA
from ComicMischiefTasks import ComicMischiefBinary, ComicMischiefMulti
import torch


class HICCAP(nn.Module):
    def __init__(self, heads=None, encoding=None, hca=None):
        super(HICCAP, self).__init__()
        if heads == None:
            raise ValueError("Heads should be either 'binary' or 'multi' or 'pretrain")     
        for head in heads:
            if head not in ['binary', 'multi', 'pretrain']:
                raise ValueError("Heads should be either 'binary' or 'multi' or 'pretrain")  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
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
                self.task_heads[head] = ComicMischiefBinary()
            elif head == "multi":
                self.task_heads[head] = ComicMischiefMulti()
            else:
                assert(head == "pretrain")
                ### Use the class that implements VTM, VAM, ATM ###
                self.task_heads[head] = ComicMischiefBinary()
            self.task_heads[head].to(device)

    def set_training_mode(self):
        self.feature_encoding.train()
        self.hca.train()
        for head in self.task_heads:
            self.task_heads[head].train()

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
        rnn_img_encoded, extended_image_attention_mask,\
            rnn_audio_encoded, extended_audio_attention_mask,\
                extended_attention_mask = self.feature_encoding(mask, 
                                                                image, 
                                                                image_mask, 
                                                                audio, 
                                                                audio_mask)
        
        output_text, output_audio, output_image = self.hca(sentences, 
                                                           rnn_img_encoded,
                                                           extended_image_attention_mask, 
                                                           rnn_audio_encoded, 
                                                           extended_audio_attention_mask, 
                                                           extended_attention_mask)
        
        return output_text, output_audio, output_image
    
    def forward_backward(self, sentences, mask, image, image_mask, audio, audio_mask,
              reg_model, actual):
        output_text, output_audio, output_image = self.forward(sentences, mask, image, 
                                                             image_mask, audio, audio_mask)
        outputs = {}
        for head in self.task_heads:
            output = self.task_heads[head].forward_backward(output_text, 
                                                        output_audio, 
                                                        output_image, 
                                                        reg_model, 
                                                        actual[head])
            outputs[head] = output
        return outputs
    
    def eval_pass(self, sentences, mask, image, image_mask, audio, audio_mask):
        output_text, output_audio, output_image = self.forward(sentences, mask, image, 
                                                               image_mask, 
                                                               audio, 
                                                               audio_mask)
        outputs = {}
        for head in self.task_heads:
            output = self.task_heads[head].eval_pass(output_text, output_audio, output_image)
            outputs[head] = output
        return outputs