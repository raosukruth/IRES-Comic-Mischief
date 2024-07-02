from torch import nn
import torch
from torch.nn import functional as F

class ComicMischiefBinary(nn.Module):
    def __init__(self):
        super(ComicMischiefBinary, self).__init__()
        self.mlp = nn.Sequential(
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
        audio_text_image  = torch.cat([output_text, output_audio, output_image], dim=-1)
        output = self.mlp(audio_text_image)
        output = F.softmax(output, dim=-1)
        return output

class ComicMischiefMulti(nn.Module):
    def __init__(self):
        super(ComicMischiefMulti, self).__init__()
        self.mature = ComicMischiefBinary()
        self.gory = ComicMischiefBinary()
        self.slapstick = ComicMischiefBinary()
        self.sarcasm = ComicMischiefBinary()

    def forward(self, output_text, output_audio, output_image, mode="all_together"):
        mature = self.mature(output_text, output_audio, output_image)
        gory = self.gory(output_text, output_audio, output_image)
        slapstick = self.slapstick(output_text, output_audio, output_image)
        sarcasm = self.sarcasm(output_text, output_audio, output_image)

        return mature, gory, slapstick, sarcasm