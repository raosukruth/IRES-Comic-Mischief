from torch import nn
import torch
from torch.nn import functional as F
import Utils

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
    
    def forward_pass(self, output_text, output_audio, output_image, reg_model, actual):
        output = self.forward(output_text, output_audio, output_image)
        y_pred = output.cpu()
        loss = Utils.compute_l2_reg_val(reg_model) + \
            F.binary_cross_entropy(y_pred, torch.Tensor(actual))
        loss.requires_grad_()
        loss.backward()
        return loss.item()
    
    def eval_pass(self, output_text, output_audio, output_image):
        out = self.forward(output_text, output_audio, output_image)
        return [out]

class ComicMischiefMulti(nn.Module):
    def __init__(self):
        super(ComicMischiefMulti, self).__init__()
        self.mature = ComicMischiefBinary()
        self.gory = ComicMischiefBinary()
        self.slapstick = ComicMischiefBinary()
        self.sarcasm = ComicMischiefBinary()
        self.mature_w = 0.1
        self.gory_w = 0.4
        self.slap_w = 0.2
        self.sarcasm_w = 0.2

    def forward(self, output_text, output_audio, output_image):
        mature = self.mature(output_text, output_audio, output_image)
        gory = self.gory(output_text, output_audio, output_image)
        slapstick = self.slapstick(output_text, output_audio, output_image)
        sarcasm = self.sarcasm(output_text, output_audio, output_image)
        return mature, gory, slapstick, sarcasm
    
    def forward_pass(self, output_text, output_audio, output_image, reg_model, actual):
        mature, gory, slapstick, sarcasm = self.forward(output_text, output_audio, output_image)
        batch_mature, batch_gory, batch_slapstick, batch_sarcasm = actual[0], actual[1], \
                                                                   actual[2], actual[3]
        mature_pred = mature.cpu()
        gory_pred = gory.cpu()
        slapstick_pred = slapstick.cpu()
        sarcasm_pred = sarcasm.cpu()
        loss1 = F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature)) 
        loss2 = F.binary_cross_entropy(gory_pred, torch.Tensor(batch_gory))
        loss3 = F.binary_cross_entropy(slapstick_pred, torch.Tensor(batch_slapstick))
        loss4 = F.binary_cross_entropy(sarcasm_pred, torch.Tensor(batch_sarcasm))

        # TODO - Implement Round Robin 
        # 
        loss = self.mature_w * loss1 + self.gory_w * loss2 + \
            self.slap_w * loss3 + self.sarcasm_w * loss4
        loss.requires_grad_()
        loss.backward()
        total_loss = loss1.item() + loss2.item() + loss3.item() + loss4.item()
        return total_loss
    
    def eval_pass(self, output_text, output_audio, output_image):
        mature, gory, slapstick, sarcasm = self.forward(output_text, output_audio, output_image)
        return [mature, gory, slapstick, sarcasm]