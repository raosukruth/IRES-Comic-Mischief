from FeatureEncoding import FeatureEncoding
from HCA import HCA
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
import FineTuning as FT

def create_encoding_hca():
    return FeatureEncoding(), HCA()

class ComicMischiefDetection:
    def __init__(self, heads=None, encoding=None, hca=None, strategy="naive", pretrain=False):
        if strategy == None:
            strategy = "naive"
        self.strategy = strategy
        if heads == None:
            raise ValueError("Heads should be either {}".format(C.supported_heads))     
        for head in heads:
            if head not in C.supported_heads:
                raise ValueError("Heads should be either {}".format(C.supported_heads))
        self.model = HICCAP(heads, encoding, hca)
        if pretrain:
            self.model.load("./fe.pth", "./hca.pth")
        self.heads = heads

    def set_training_mode(self):
        self.model.set_training_mode()

    def set_eval_mode(self):
        self.model.set_eval_mode()  

    def training_loop(self, start_epoch, max_epochs, 
                      train_set, validation_set, 
                      optimizer_type="adam"):
        learning_rate = 1.9e-5

        params = self.model.get_model_params()
        if optimizer_type == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=0.01)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=0.01)
        
        strategy = None
        if self.strategy == "naive":
            strategy = FT.Naive(self.heads)
        elif self.strategy == "weighted":
            strategy = FT.Weighted(self.heads)
        else:
            assert(self.strategy == "dsg")
            strategy = FT.DynamicStopAndGo(self.heads)
        assert(strategy != None)
        loss_history = {
            "binary": [],
            "mature": [],
            "gory": [],
            "slapstick": [],
            "sarcasm": []
        }
        for _ in range(start_epoch, max_epochs):
            self.train(train_set, validation_set, optimizer, strategy, loss_history)
            avg_loss, accuracy, f1 = self.evaluate(validation_set)
            for head in avg_loss:
                print("Validation {}: avg_loss = {:.4f}; accuracy = {:.4f}; f1 = {:.4f}".format(
                    head, avg_loss[head], accuracy[head], f1[head]))
            Utils.save_dict("{}_validation_avg_loss.pkl".format(self.strategy), avg_loss)
            Utils.save_dict("{}_validation_accuracy.pkl".format(self.strategy), accuracy)
            Utils.save_dict("{}_validation_f1.pkl".format(self.strategy), f1)

        Utils.save_dict("{}_train_loss_history.pkl".format(self.strategy), loss_history)
 
    def train(self, training_set, validation_set, optimizer, strategy, 
              loss_history,
              batch_size=C.batch_size,
              text_pad_length=500, img_pad_length=36, 
              audio_pad_length=63, shuffle=True, 
              device=None):
        if device == None:
            device = C.device
        
        dataset = CustomDataset(training_set, text_pad_length, 
                                img_pad_length, audio_pad_length)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle)
        self.set_training_mode()
        eval_batch_count = strategy.get_eval_iter_count()
        for batch_idx, batch in enumerate(dataloader):
            strategy.start_iter()
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
                elif head == "mature":
                    actual[head] = batch_mature
                elif head == "gory":
                    actual[head] = batch_gory
                elif head == "slapstick":
                    actual[head] = batch_slapstick
                else:
                    assert(head == "sarcasm")
                    actual[head] = batch_sarcasm

            outputs = self.model.forward_pass(batch_text, 
                                              batch_text_mask, 
                                              batch_image, 
                                              batch_mask_img, 
                                              batch_audio, 
                                              batch_mask_audio,
                                              actual)
            Utils.update_loss_history(loss_history, outputs)
            optimizer.zero_grad()
            strategy.backward(outputs)
            optimizer.step()

            if C.show_training_loss:
                for head, loss in outputs.items():
                    print("Training Batch: {}, Head: {}, Loss: {}".format(
                        batch_idx, head, loss.item()))
                print("\n")

            if (batch_idx + 1) % eval_batch_count == 0:
                loss, accuracy, _ = self.evaluate(validation_set, is_training=True)
                strategy.process_eval(loss, accuracy)
            strategy.end_iter()

    def evaluate(self, json_data, is_training=False,
                 batch_size=C.batch_size, text_pad_length=500, 
                 img_pad_length=36, audio_pad_length=63,
                 shuffle=True, device=None):
        if not is_training:
            self.set_eval_mode()
        if device == None:
            device = C.device

        dataset = CustomDataset(json_data, text_pad_length, 
                                img_pad_length, audio_pad_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.set_eval_mode()

        total_loss = {} 
        all_preds = {}
        all_labels = {}
        
        for batch_idx, batch in enumerate(dataloader):
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
                elif head == "mature":
                    pred = batch["mature"].to(device) # batch_size by 2
                    batch_pred = [pred]
                elif head == "gory":
                    pred = batch["gory"].to(device) # batch_size by 2
                    batch_pred = [pred]
                elif head == "slapstick":
                    pred = batch["slapstick"].to(device) # batch_size by 2
                    batch_pred = [pred]
                else:
                    assert(head == "sarcasm")
                    pred = batch["sarcasm"].to(device) # batch_size by 2
                    batch_pred = [pred]
                output = [output]
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
        Utils.save_dict("{}_train_avg_loss.pkl".format(self.strategy), avg_loss)
        Utils.save_dict("{}_train_accuracy.pkl".format(self.strategy), accuracy)
        Utils.save_dict("{}_train_f1.pkl".format(self.strategy), f1)