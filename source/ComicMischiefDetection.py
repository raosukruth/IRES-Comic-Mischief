from FeatureEncoding import FeatureEncoding
from HCA import HCA
from ComicMischiefTasks import ComicMischiefBinary, ComicMischiefMulti
import torch
from torch.nn import functional as F
import numpy as np
import os
import config as C
import json
from torch import optim

class ComicMischiefDetection:
    def __init__(self, head="binary"):
        self.feature_encoding = FeatureEncoding()
        self.hca = HCA()
        if head == "binary":
            self.task_head = ComicMischiefBinary()
        elif head == "multi":
            self.task_head = ComicMischiefMulti()
        else:
            ### Use the class that implements VTM, VAM, ATM ###
            self.task_head = None

    def set_training_mode(self):
         self.feature_encoding.train()
         self.hca.train()
         self.task_head.train()

    def compute_l2_reg_val(self, model):
        l2_lambda = 0.1
        l2_reg = None
        for w in model.parameters():
            if l2_reg is None:
                l2_reg = w.norm(2)
            else:
                l2_reg = l2_reg + w.norm(2)
        return l2_lambda * l2_reg.item()
    
    def masking(self, docs_ints, seq_length=500):
        masks = np.zeros((len(docs_ints), seq_length), dtype=int)
        for i, row in enumerate(docs_ints):
            masks[i, -len(row):] = 1
        return masks
    
    def pad_segment(self, feature, max_feature_len):
        S, D = feature.shape
        try:
            pad_l =  max_feature_len - S
            # pad
            pad_segment = np.zeros((pad_l, D))
            feature = np.concatenate((feature,pad_segment), axis=0)
        except:
            feature = feature[0:max_feature_len]
        return feature
    
    def mask_vector(self, max_size, arr):
        if arr.shape[0] > max_size:
            output = [1]*max_size
        else:
            len_zero_value = max_size -  arr.shape[0]
            output = [1]*arr.shape[0] + [0]*len_zero_value
        return np.array(output)

    def training_loop(self, start_epoch, max_epochs, optimizer_type="adam"):
        model = self.task_head
        learning_rate = 1.9e-5
        weight_decay_val = 0

        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay_val)
        for _ in range(start_epoch, max_epochs):
            ### Improve this ###
            self.train(optimizer)

       

    def train(self, optimizer):
        """
        Trains the model using the optimizer for a single epoch.
        :param model: pytorch model
        :param optimizer:
        :return:
        """

        self.set_training_mode()

        batch_idx = 1
        total_loss = 0
        batch_x = []
        batch_image, batch_mask_img = [],[]
        batch_audio, batch_mask_audio = [],[]
        batch_y = []
        train_imdb = []
        features_dict_train = json.load(open(C.training_features))
        train_set = features_dict_train
        sh_train_set = train_set
        
        for index, i in enumerate(sh_train_set):
            #list(np.int_(batch_x))
            mid = sh_train_set[i]['IMDBid']

            file_path = "path_to_I3D_features/"+mid+"_rgb.npy"
            if not os.path.isfile(file_path): 
                print(index, "  - mid:", mid)
                continue
            if mid == "laqIl3LniQE.02":
                    a1 = np.array([1024*[0.0]])
                    a2 = np.array([1024*[0.0]])
                    continue
            else:
                    path = "path_to_I3D_features/"
                    #image_vec = np.load("./deepMoji_out/"+mid+".npy")
                    a1 = np.load(path+mid+"_rgb.npy")
                    a2 = np.load(path+mid+"_flow.npy")
            a = a1+a2
            masked_img = self.mask_vector(36,a)
            a = self.pad_segment(a, 36, 0)
            image_vec = a

            path = "path_to_VGGish_features/"
            try:
                audio_arr = np.load(path+mid+"_vggish.npy")
            except:
                audio_arr = np.array([128*[0.0]])
            masked_audio = self.mask_vector(63,audio_arr)
            audio_vec = self.pad_segment(audio_arr, 63, 0)
            batch_audio.append(audio_vec)
            batch_mask_audio.append(masked_audio)

            train_imdb.append(mid)
            batch_x.append(np.array(sh_train_set[i]['indexes']))
            batch_mask_img.append(masked_img)
            batch_image.append(image_vec)
            batch_y.append(sh_train_set[i]['y'])
            
            batch_size = C.batch_size
            if (len(batch_x) == batch_size or index == len(sh_train_set) - 1 ) and len(batch_x) > 0:
                optimizer.zero_grad()

                mask = self.masking(batch_x)
                batch_x = self.pad_features(batch_x)
                batch_x = np.array(batch_x)
                batch_x = torch.tensor(batch_x).cuda()

                batch_image = np.array(batch_image)
                batch_image = torch.tensor(batch_image).cuda()

                batch_mask_img = np.array(batch_mask_img)
                batch_mask_img = torch.tensor(batch_mask_img).cuda()

                batch_audio = np.array(batch_audio)
                batch_audio = torch.tensor(batch_audio).cuda()

                batch_mask_audio = np.array(batch_mask_audio)
                batch_mask_audio = torch.tensor(batch_mask_audio).cuda()

                self.feature_encoding(torch.tensor(mask).cuda(), 
                                        batch_image.float(), 
                                        batch_mask_img,
                                        batch_audio,
                                        batch_mask_audio)

                self.hca(batch_x, 
                            self.feature_encoding.rnn_img_encoded, 
                            self.feature_encoding.extended_image_attention_mask,
                            self.feature_encoding.rnn_audio_encoded, 
                            self.feature_encoding.extended_audio_attention_mask,
                            self.feature_encoding.extended_attention_mask
                        )

                out, mid = self.task_head(self.hca.output_text, self.hca.output_audio, self.hca.output_image)
                y_pred1 = out.cpu()
                loss2 = self.compute_l2_reg_val(self.task_head) + F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y))
                
                total_loss += loss2.item()

                loss2.backward()


                optimizer.step()

                ### Need to fix ###
                #torch_helper = TorchHelper()

                #torch_helper.show_progress(batch_idx, np.ceil(len(sh_train_set) / batch_size), start_time,
                #                        round(total_loss / (index + 1), 4))


                batch_idx += 1
                batch_x, batch_y,batch_image,batch_mask_img = [], [], [],[]
                batch_audio, batch_mask_audio = [],[]