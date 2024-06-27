from torch import nn
import torch
from torch.nn import functional as F

class ComicMischiefBinary(nn.Module):
    def __init__(self):
        super(ComicMischiefBinary, self).__init__()
        self.img_audio_text_linear = nn.Sequential(
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
        output = F.softmax(self.img_audio_text_linear(audio_text_image), -1)
        return output, []

class ComicMischiefMulti:
    def __init__(self):
        pass

    def train(self):
        pass