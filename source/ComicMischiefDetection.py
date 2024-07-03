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

def create_encoding_hca():
    return FeatureEncoding(), HCA()

class ComicMischiefDetection:
    def __init__(self, heads=None, encoding=None, hca=None):
        if heads == None:
            raise ValueError("Heads should be either 'binary' or 'multi' or 'pretrain")     
        for head in heads:
            if head not in ['binary', 'multi', 'pretrain']:
                raise ValueError("Heads should be either 'binary' or 'multi' or 'pretrain")     
        self.model = HICCAP(heads, encoding, hca)
        self.heads = heads

    def set_training_mode(self):
        self.model.set_training_mode()

    def set_eval_mode(self):
        self.model.set_eval_mode()  

    def training_loop(self, start_epoch, max_epochs, 
                      train_set, validation_set, 
                      optimizer_type="adam", pretrain=False):
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
            self.train(train_set, optimizer, pretrain=pretrain)
            avg_loss, accuracy, f1 = self.evaluate(validation_set)
            for head in avg_loss:
                print("Validation {}: avg_loss = {:.4f}; accuracy = {:.4f}; f1 = {:.4f}".format(
                    head, avg_loss[head], accuracy[head], f1[head]))
            if lr_schedule_active:
                lr_scheduler.step(f1)
    
    def train(self, json_data, optimizer, batch_size=24,
              text_pad_length=500, img_pad_length=36, 
              audio_pad_length=63, shuffle=True, 
              device=None, pretrain=False):
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu" # hack to overcome cuda running out of memory
        
        dataset = CustomDataset(json_data, text_pad_length, 
                                img_pad_length, audio_pad_length)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle)
        self.set_training_mode()

        total_loss = {}
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
                actual = {}
                for head in self.heads:
                    if head == "binary":
                        actual[head] = batch_binary
                    elif head == "multi":
                        actual[head] = [batch_mature, batch_gory, 
                                        batch_slapstick, batch_sarcasm]
                    else:
                        ### HACk for Pretrain
                        assert(head == "pretrain")
                        actual[head] = batch_binary

                outputs = self.model.forward_backward(batch_text, 
                                                      batch_text_mask, 
                                                      batch_image, 
                                                      batch_mask_img, 
                                                      batch_audio, 
                                                      batch_mask_audio,
                                                      self.model,
                                                      actual)
                for head, loss in outputs.items():
                    if head not in total_loss:
                        total_loss[head] = 0
                    total_loss[head] += loss
                optimizer.step()
                batch_idx += 1
                break

    def evaluate(self, json_data, batch_size=24, 
                 text_pad_length=500, img_pad_length=36, 
                 audio_pad_length=63, shuffle=True, 
                 device=None):
        self.set_eval_mode()
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu" # Hack to overcome CUDA out of memory condtion 

        dataset = CustomDataset(json_data, text_pad_length, 
                                img_pad_length, audio_pad_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.set_eval_mode()

        total_loss = {} 
        all_preds = {}
        all_labels = {}
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
                outputs = self.model.eval_pass(batch_text, 
                                           batch_text_mask, 
                                           batch_image, 
                                           batch_mask_img, 
                                           batch_audio, 
                                           batch_mask_audio)
                for head, output in outputs.items():
                    if head == "binary":
                        pred = batch["binary"].to(device) # batch_size by 2
                        batch_pred = [pred]
                    elif head == "multi":
                        batch_pred = [batch_mature, batch_gory, 
                                      batch_slapstick, batch_sarcasm]
                    else:
                        assert(head == "pretrain")
                        ### Hack FOR PRETRAINING ###
                        pred = batch["binary"].to(device) # batch_size by 2
                        batch_pred = [pred]

                    for out, pred in zip(output, batch_pred):
                        loss = F.binary_cross_entropy(out, pred)
                        if head not in total_loss:
                            total_loss[head] = 0
                        total_loss[head] += loss.item()

                        # Collect predictions and true labels
                        preds = (out[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                        true_labels = (pred[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                    
                        if head not in all_preds:
                            all_preds[head] = []
                        all_preds[head].extend(preds)
                        if head not in all_labels:
                            all_labels[head] = []
                        all_labels[head].extend(true_labels)
                break
        accuracy = {}
        f1 = {}
        avg_loss = {}
        for head, labels in all_labels.items():
            # Calculate accuracy and F1 score
            accuracy[head] = accuracy_score(labels, all_preds[head])
            f1[head] = f1_score(labels, all_preds[head], 
                                average='macro')  # use 'macro' or 'weighted' for multi-class
            avg_loss[head] = total_loss[head] / len(dataloader)
        return avg_loss, accuracy, f1

    def test(self):
        avg_loss, accuracy, f1 = self.evaluate("test_features_lrec_camera.json")
        for head in avg_loss:
            print("Test {}: avg_loss = {:.4f}; accuracy = {:.4f}; f1 = {:.4f}".format(
                head, avg_loss[head], accuracy[head], f1[head]))