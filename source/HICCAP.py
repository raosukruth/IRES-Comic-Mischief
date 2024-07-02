from torch import nn
from FeatureEncoding import FeatureEncoding
from HCA import HCA
from ComicMischiefTasks import ComicMischiefBinary, ComicMischiefMulti
import torch


class HICCAP(nn.Module):
    def __init__(self, head="binary", encoding=None, hca=None):
        super(HICCAP, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if encoding == None:
            encoding = FeatureEncoding()
        if hca == None:
            hca = HCA()
        self.feature_encoding = encoding
        self.hca = hca

        self.feature_encoding.to(device)
        self.hca.to(device)

        if head == "binary":
            self.task_head = ComicMischiefBinary()
        elif head == "multi":
            self.task_head = ComicMischiefMulti()
        else:
            ### Use the class that implements VTM, VAM, ATM ###
            self.task_head = None

        self.task_head.to(device)


    def set_training_mode(self):
        self.feature_encoding.train()
        self.hca.train()
        self.task_head.train()

    def set_eval_mode(self):
        self.feature_encoding.eval()
        self.hca.eval()
        self.task_head.eval()

    def get_model_params(self):
        params = [{'params': self.feature_encoding.parameters()},
                 {'params': self.hca.parameters()},
                  {'params': self.task_head.parameters()}]
        
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
        
        out = self.task_head(output_text, output_audio, output_image)

        return out


    