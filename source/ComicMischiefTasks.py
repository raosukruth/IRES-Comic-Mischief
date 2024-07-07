from torch import nn
import torch
from torch.nn import functional as F
import Utils

class ComicMischiefBinary(nn.Module):
    def __init__(self):
        super(ComicMischiefBinary, self).__init__()
        self.binary = nn.Sequential(
            nn.Linear(768*3  , 200),
            nn.BatchNorm1d(200),
            nn.Dropout(0.3),

            nn.Linear(200, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(0.3),

            nn.Linear(40, 20),
            nn.Linear(20, 2)
        )

    def forward(self, output_text, output_audio, output_image):
        audio_text_image  = torch.cat([output_text, output_audio, 
                                       output_image], dim=-1)
        output = self.binary(audio_text_image)
        output = F.softmax(output, dim=-1)
        return output
    
    def forward_pass(self, output_text, output_audio, output_image, actual):
        y_pred = self.forward(output_text, output_audio, output_image)
        loss = F.binary_cross_entropy(y_pred, torch.Tensor(actual))
        return {
            'binary': loss
        }
    
    def eval_pass(self, output_text, output_audio, output_image):
        out = self.forward(output_text, output_audio, output_image)
        return {
            'binary': out
        }

class ComicMischiefMature(nn.Module):
    def __init__(self):
        super(ComicMischiefMature, self).__init__()
        self.mature = ComicMischiefBinary()

    def forward(self, output_text, output_audio, output_image):
        mature = self.mature(output_text, output_audio, output_image)
        return mature
    
    def forward_pass(self, output_text, output_audio, 
                     output_image, actual):
        mature_pred = self.forward(output_text, 
                              output_audio, 
                              output_image)
        batch_mature = actual
        loss = F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature)) 
        return {
            'mature': loss,
        }
    
    def eval_pass(self, output_text, output_audio, output_image):
        mature = self.forward(output_text, 
                              output_audio, 
                              output_image)
        return {
            'mature': mature, 
        }

class ComicMischiefGory(nn.Module):
    def __init__(self):
        super(ComicMischiefGory, self).__init__()
        self.gory = ComicMischiefBinary()

    def forward(self, output_text, output_audio, output_image):
        gory = self.gory(output_text, output_audio, output_image)
        return gory
    
    def forward_pass(self, output_text, output_audio, 
                     output_image, actual):
        gory_pred = self.forward(output_text, 
                            output_audio, 
                            output_image)
        batch_gory = actual
        loss = F.binary_cross_entropy(gory_pred, torch.Tensor(batch_gory))
        return {
            'gory': loss,
        }
    
    def eval_pass(self, output_text, output_audio, output_image):
        gory = self.forward(output_text, 
                            output_audio, 
                            output_image)
        return {
            'gory': gory, 
        }

class ComicMischiefSlapstick(nn.Module):
    def __init__(self):
        super(ComicMischiefSlapstick, self).__init__()
        self.slapstick = ComicMischiefBinary()

    def forward(self, output_text, output_audio, output_image):
        slapstick = self.slapstick(output_text, output_audio, output_image)
        return slapstick
        
    def forward_pass(self, output_text, output_audio, 
                     output_image, actual):
        slapstick_pred = self.forward(output_text, 
                                 output_audio, 
                                 output_image)
        batch_slapstick = actual
        loss = F.binary_cross_entropy(slapstick_pred, torch.Tensor(batch_slapstick))
        return {
            'slapstick': loss,
        }
    
    def eval_pass(self, output_text, output_audio, output_image):
        slapstick = self.forward(output_text, 
                                 output_audio, 
                                 output_image)
        return {
            'slapstick': slapstick, 
        }

class ComicMischiefSarcasm(nn.Module):
    def __init__(self):
        super(ComicMischiefSarcasm, self).__init__()
        self.sarcasm = ComicMischiefBinary()

    def forward(self, output_text, output_audio, output_image):
        sarcasm = self.sarcasm(output_text, output_audio, output_image)
        return sarcasm
    
    def forward_pass(self, output_text, output_audio, 
                     output_image, actual):
        sarcasm_pred = self.forward(output_text, 
                               output_audio, 
                               output_image)
        batch_sarcasm = actual
        loss = F.binary_cross_entropy(sarcasm_pred, torch.Tensor(batch_sarcasm))
        return {
            'sarcasm': loss
        }
    
    def eval_pass(self, output_text, output_audio, output_image):
        sarcasm = self.forward(output_text, 
                               output_audio, 
                               output_image)
        return {
            'sarcasm': sarcasm
        }