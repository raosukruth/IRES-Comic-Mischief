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
            ### Improve this ###
            self.train(optimizer, train_set)

            avg_loss, accuracy, f1 = self.evaluate(validation_set)
            print("Validation {}: avg_loss = {:.4f}; accuracy = {:.4f}; f1 = {:.4f}".format(
                self.head, avg_loss, accuracy, f1))
            
            if lr_schedule_active:
                lr_scheduler.step(f1)

    def train(self, optimizer, dataset):
 

        global mature_w, gory_w, slap_w, sarcasm_w

        self.set_training_mode()

        batch_size = C.batch_size
        batch_idx = 1
        total_loss = 0
        batch_x = []
        batch_image, batch_mask_img = [], []
        batch_audio, batch_mask_audio = [], []
        batch_mask = []
        batch_y = []
        train_imdb = []
        sh_train_set = dataset

        if self.head == 'multi':
            batch_mature, batch_gory, batch_slapstick, batch_sarcasm = [], [], [], []
            # log_vars = nn.Parameter(torch.zeros((4)))
        
        for index, i in enumerate(sh_train_set):
            mid = sh_train_set[i]['IMDBid']
            
            if self.head == 'multi' and sh_train_set[i]['label'] == 0:
                continue

            file_path = C.path_to_I3D_features + mid + "_rgb.npy"
            if not os.path.isfile(file_path): 
                print(index, "  - mid:", mid)
                continue

            path = C.path_to_I3D_features
            a1 = np.load(path + mid + "_rgb.npy")
            a2 = np.load(path + mid + "_flow.npy")
            a = a1 + a2
            masked_img = Utils.mask_vector(36, a)
            a = Utils.pad_segment(a, 36, 0)
            image_vec = a

            path = C.path_to_VGGish_features
            try:
                audio_arr = np.load(path + mid + "_vggish.npy")
            except:
                audio_arr = np.array([128 * [0.0]])
            masked_audio = Utils.mask_vector(63, audio_arr)
            audio_vec = Utils.pad_segment(audio_arr, 63, 0)

            batch_audio.append(audio_vec)
            batch_mask_audio.append(masked_audio)
            train_imdb.append(mid)
            batch_x.append(np.array(sh_train_set[i]['indexes']))
            batch_mask_img.append(masked_img)
            batch_image.append(image_vec)

            if self.head == 'binary':
                batch_y.append(sh_train_set[i]['y'])
            elif self.head == 'multi':
                batch_mature.append(sh_train_set[i]['mature'])
                batch_gory.append(sh_train_set[i]['gory'])
                batch_slapstick.append(sh_train_set[i]['slapstick'])
                batch_sarcasm.append(sh_train_set[i]['sarcasm'])

            if (len(batch_x) == batch_size or index == len(sh_train_set) - 1) and len(batch_x) > 0:
                optimizer.zero_grad()
                import pdb;pdb.set_trace()

                mask = Utils.masking(batch_x)
                batch_x = Utils.pad_features(batch_x)
                batch_x = torch.tensor(batch_x).cuda()
                batch_image = torch.cat(batch_image, dim=0).cuda()
                batch_mask_img = torch.cat(batch_mask_img, dim=0).cuda()
                batch_audio = torch.cat(batch_audio, dim=0).cuda()
                batch_mask_audio = torch.cat(batch_mask_audio, dim=0).cuda()
                
                out = self.model(batch_x, 
                                      torch.tensor(mask).cuda(), 
                                      batch_image.float(), 
                                      batch_mask_img, 
                                      batch_audio.float(), 
                                      batch_mask_audio)

                if self.head == 'binary':
                    y_pred1 = out.cpu()
                    loss2 = Utils.compute_l2_reg_val(self.task_head) + \
                        F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y))
                    total_loss += loss2.item()
                    loss2.backward()

                elif self.head == 'multi':
                    mature_pred = out[0].cpu()
                    gory_pred = out[1].cpu()
                    slapstick_pred = out[2].cpu()
                    sarcasm_pred = out[3].cpu()
                    loss1 = F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature)) 
                    loss2 = F.binary_cross_entropy(gory_pred, torch.Tensor(batch_gory))
                    loss3 = F.binary_cross_entropy(slapstick_pred, torch.Tensor(batch_slapstick))
                    loss4 = F.binary_cross_entropy(sarcasm_pred, torch.Tensor(batch_sarcasm))
                    total_loss += loss1.item() + loss2.item() + loss3.item() + loss4.item()
                    loss = mature_w * loss1 + gory_w * loss2 + slap_w * loss3 + sarcasm_w * loss4
                    loss.backward()

                optimizer.step()

                # torch_helper.show_progress(batch_idx, np.ceil(len(sh_train_set) / batch_size), start_time, round(total_loss / (index + 1), 4))
                batch_idx += 1
                batch_x, batch_image, batch_mask_img, batch_audio, batch_mask_audio = [], [], [], [], []
                if self.head == 'binary':
                    batch_y = []
                elif self.head == 'multi':
                    batch_mature, batch_gory, batch_slapstick, batch_sarcasm = [], [], [], []

    def evaluate_binary(self, json_data, batch_size=24, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
        self.set_eval_mode()

        total_loss = 0 
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_text = batch['text'].to(device)
                batch_text_mask = batch['text_mask'].to(device)
                batch_image = batch['image'].float().to(device)
                batch_mask_img = batch['image_mask'].to(device)
                batch_audio = batch['audio'].float().to(device)
                batch_mask_audio = batch['audio_mask'].to(device)
                batch_pred = batch["binary"].to(device) # batch_size by 2

                # batch_size by 2 for binary
                # batch size by 4 by 2 for multi task
                out = self.model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio)
                loss = F.binary_cross_entropy(out, batch_pred)
                total_loss += loss.item()

                # Collect predictions and true labels
                preds = (out[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                true_labels = (batch_pred[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                
                all_preds.extend(preds)
                all_labels.extend(true_labels)

                if batch_idx == 20:
                    break
        
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')  # use 'macro' or 'weighted' for multi-class

        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy, f1


    def evaluate_multi(self, json_data, batch_size=24, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
        self.set_eval_mode()

        total_loss = 0 
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_text = batch['text'].to(device)
                batch_text_mask = batch['text_mask'].to(device)
                batch_image = batch['image'].float().to(device)
                batch_mask_img = batch['image_mask'].to(device)
                batch_audio = batch['audio'].float().to(device)
                batch_mask_audio = batch['audio_mask'].to(device)
                batch_pred = batch["binary"].to(device) # batch_size by 2

                # batch_size by 2 for binary
                # batch size by 4 by 2 for multi task
                mature, gory, slapstick, sarcasm = self.model(batch_text, 
                                                              batch_text_mask, 
                                                              batch_image, 
                                                              batch_mask_img, 
                                                              batch_audio, 
                                                              batch_mask_audio)
                outputs = [mature, gory, slapstick, sarcasm]

                for out in outputs:
                    loss = F.binary_cross_entropy(out, batch_pred)
                    total_loss += loss.item()

                    # Collect predictions and true labels
                    preds = (out[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                    true_labels = (batch_pred[:, 1] > 0.5).cpu().numpy()  # Using the second column for binary classification
                
                    all_preds.extend(preds)
                    all_labels.extend(true_labels)

                if batch_idx == 20:
                    break
    
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')  # use 'macro' or 'weighted' for multi-class
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy, f1


    def evaluate(self, json_data):
        self.set_eval_mode()

        if self.head == "binary":
            return self.evaluate_binary(json_data)
        
        return self.evaluate_multi(json_data)
            
    def test(self):
        avg_loss, accuracy, f1 = self.evaluate("test_features_lrec_camera.json")
        print("Test {}: avg_loss = {:.4f}; accuracy = {:.4f}; f1 = {:.4f}".format(
            self.head, avg_loss, accuracy, f1))