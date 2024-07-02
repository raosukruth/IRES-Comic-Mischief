from FeatureEncoding import FeatureEncoding
from HCA import HCA
from ComicMischiefTasks import ComicMischiefBinary, ComicMischiefMulti
import torch
from torch.nn import functional as F
import numpy as np
import os
import config as C
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report
from HICCAP import HICCAP
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
import Utils
import torch.nn as nn

def create_encoding_hca():
    return FeatureEncoding(), HCA()

class ComicMischiefDetection:
    def __init__(self, head="binary", encoding=None, hca=None):
        self.model = HICCAP(head, encoding, hca)
        self.head = head
        if self.head not in ['binary', 'multi']:
            raise ValueError("Mode should be either 'binary' or 'multi'")     

    def set_training_mode(self):
        self.model.set_training_mode()

    def set_eval_mode(self):
        self.model.set_eval_mode()  

    def training_loop(self, start_epoch, max_epochs, 
                      train_set, validation_set, 
                      optimizer_type="adam"):
        learning_rate = 1.9e-5
        weight_decay_val = 0
        lr_schedule_active = False
        reduce_on_plateau_lr_schdlr = \
            torch.optim.lr_scheduler.ReduceLROnPlateau

        params = self.model.get_model_params()
        if optimizer_type == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate)
        
        lr_scheduler = reduce_on_plateau_lr_schdlr(optimizer, 
                                                   'max', 
                                                   min_lr=1e-8,
                                                   patience=2, 
                                                   factor=0.5)

        for _ in range(start_epoch, max_epochs):
            self.train(train_set, optimizer)
            avg_loss, accuracy, f1 = self.evaluate(validation_set)
            print("Validation {}: avg_loss = {:.4f}; accuracy = {:.4f}; f1 = {:.4f}".format(
                self.head, avg_loss, accuracy, f1))
            if lr_schedule_active:
                lr_scheduler.step(f1)
    
    def train(self, json_data, optimizer, batch_size=24,
              text_pad_length=500, img_pad_length=36, 
              audio_pad_length=63, shuffle=True, 
              device=None):
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu" # hack to overcome cuda running out of memory
        dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.set_eval_mode()

        mature_w = 0.1
        gory_w = 0.4
        slap_w = 0.2
        sarcasm_w = 0.2
        
        total_loss = 0 
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()

                batch_text = batch['text'].to(device)
                batch_text_mask = batch['text_mask'].to(device)
                batch_image = batch['image'].float().to(device)
                batch_mask_img = batch['image_mask'].to(device)
                batch_audio = batch['audio'].float().to(device)
                batch_mask_audio = batch['audio_mask'].to(device)
                batch_binary = batch["binary"].to(device) # batch_size by 2
                batch_mature = batch["mature"].to(device)
                batch_gory = batch["gory"].to(device)
                batch_slapstick = batch["slapstick"].to(device)
                batch_sarcasm = batch["sarcasm"].to(device)

                # batch_size by 2 for binary
                # batch size by 4 by 2 for multi task
                if self.head == "binary":
                    out = self.model(batch_text, 
                                    batch_text_mask, 
                                    batch_image, 
                                    batch_mask_img, 
                                    batch_audio, 
                                    batch_mask_audio)

                    y_pred = out.cpu()
                    
                    loss = Utils.compute_l2_reg_val(self.model) + \
                        F.binary_cross_entropy(y_pred, torch.Tensor(batch_binary))
                    total_loss += loss.item()
                else:
                    mature, gory, slapstick, sarcasm = self.model(batch_text, 
                                                batch_text_mask, 
                                                batch_image, 
                                                batch_mask_img, 
                                                batch_audio, 
                                                batch_mask_audio)
                    
                    mature_pred = mature.cpu()
                    gory_pred = gory.cpu()
                    slapstick_pred = slapstick.cpu()
                    sarcasm_pred = sarcasm.cpu()
                    loss1 = F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature)) 
                    loss2 = F.binary_cross_entropy(gory_pred, torch.Tensor(batch_gory))
                    loss3 = F.binary_cross_entropy(slapstick_pred, torch.Tensor(batch_slapstick))
                    loss4 = F.binary_cross_entropy(sarcasm_pred, torch.Tensor(batch_sarcasm))
                    total_loss += loss1.item() + loss2.item() + loss3.item() + loss4.item()
                    loss = mature_w * loss1 + gory_w * loss2 + slap_w * loss3 + sarcasm_w * loss4

                loss.requires_grad_()
                loss.backward()
                optimizer.step()
                batch_idx += 1

    def evaluate(self, json_data, batch_size=24, 
                 text_pad_length=500, img_pad_length=36, 
                 audio_pad_length=63, shuffle=True, 
                 device=None):
        self.set_eval_mode()
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu" # Hack to overcome CUDA out of memory condtion 

        dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.set_eval_mode()

        total_loss = 0 
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                batch_text = batch['text'].to(device)
                batch_text_mask = batch['text_mask'].to(device)
                batch_image = batch['image'].float().to(device)
                batch_mask_img = batch['image_mask'].to(device)
                batch_audio = batch['audio'].float().to(device)
                batch_mask_audio = batch['audio_mask'].to(device)
                batch_mature = batch["mature"].to(device) # batch_size by 2
                batch_gory = batch["gory"].to(device) # batch_size by 2
                batch_slapstick = batch["slapstick"].to(device) # batch_size by 2
                batch_sarcasm = batch["sarcasm"].to(device) # batch_size by 2
                if self.head == "binary":
                    pred = batch["binary"].to(device) # batch_size by 2
                    batch_pred = [pred]
                    out = self.model(batch_text, 
                                     batch_text_mask, 
                                     batch_image, 
                                     batch_mask_img, 
                                     batch_audio, 
                                     batch_mask_audio)
                    outputs = [out]
                elif self.head == "multi":
                    batch_pred = [batch_mature, batch_gory, batch_slapstick, batch_sarcasm]
                    # batch_size by 2 for binary
                    # batch size by 4 by 2 for multi task
                    mature, gory, slapstick, sarcasm = self.model(batch_text,
                                                                  batch_text_mask, 
                                                                  batch_image, 
                                                                  batch_mask_img, 
                                                                  batch_audio, 
                                                                  batch_mask_audio)
                    outputs = [mature, gory, slapstick, sarcasm]
                else:
                    ### CODE FOR PRETRAINING ###
                    pass

                for out, pred in zip(outputs, batch_pred):
                    loss = F.binary_cross_entropy(out, pred)
                    total_loss += loss.item()

                    # Collect predictions and true labels
                    preds = (out[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                    true_labels = (pred[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                
                    all_preds.extend(preds)
                    all_labels.extend(true_labels)

        # Calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')  # use 'macro' or 'weighted' for multi-class
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy, f1

    def test(self):
        avg_loss, accuracy, f1 = self.evaluate("test_features_lrec_camera.json")
        print("Test {}: avg_loss = {:.4f}; accuracy = {:.4f}; f1 = {:.4f}".format(
            self.head, avg_loss, accuracy, f1))