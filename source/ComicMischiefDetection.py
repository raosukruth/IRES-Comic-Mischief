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

    def set_training_mode(self):
        self.model.set_training_mode()

    def set_eval_mode(self):
        self.model.set_eval_mode()

    # def compute_l2_reg_val(self, model):
    #     l2_lambda = 0.1
    #     l2_reg = None
    #     for w in model.parameters():
    #         if l2_reg is None:
    #             l2_reg = w.norm(2)
    #         else:
    #             l2_reg = l2_reg + w.norm(2)
    #     return l2_lambda * l2_reg.item()
    
    # def masking(self, docs_ints, seq_length=500):
    #     masks = np.zeros((len(docs_ints), seq_length), dtype=int)
    #     for i, row in enumerate(docs_ints):
    #         masks[i, -len(row):] = 1
    #     return masks
    
    # # def pad_segment(self, feature, max_feature_len):
    # #     S, D = feature.shape
    # #     try:
    # #         pad_l =  max_feature_len - S
    # #         # pad
    # #         pad_segment = np.zeros((pad_l, D))
    # #         feature = np.concatenate((feature,pad_segment), axis=0)
    # #     except:
    # #         feature = feature[0:max_feature_len]
    # #     return feature


    # def pad_segment(self, feature, max_feature_len, pad_idx):
    #     S, D = feature.shape
    #     if S > max_feature_len:
    #         feature = feature[:max_feature_len]
    #     else:
    #         pad_l = max_feature_len - S
    #         pad_segment = torch.zeros((pad_l, D))
    #         feature = torch.concatenate((feature, pad_segment), axis=0)
    #     return feature

    
    # def mask_vector(self, max_size, arr):
    #     if arr.shape[0] > max_size:
    #         output = [1]*max_size
    #     else:
    #         len_zero_value = max_size -  arr.shape[0]
    #         output = [1]*arr.shape[0] + [0]*len_zero_value
    #     return np.array(output)
    
    # def hamming_score(self, y_true, y_pred, 
    #                   normalize=True, sample_weight=None):
    #     acc_list = []
    #     for i in range(y_true.shape[0]):
    #         c = 0
    #         for j in range(len(y_true[i])):
    #             if y_true[i][j] == y_pred[i][j]:
    #                 c += 1
    #         acc_list.append (c/4)
    #     return np.mean(acc_list)

    def training_loop(self, start_epoch, max_epochs, 
                      mode, train_set, validation_set, 
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
            self.train(optimizer, train_set, mode)

            if mode == "binary":
                print(self.evaluate(mode, validation_set, "val"))
                # _, _, _, _, _, _, val_binary = self.evaluate(mode, validation_set, "val")
                # _, _, _
                # if lr_schedule_active:
                #     lr_scheduler.step(val_binary)
            
            # else:
            #     _, _, _, _, _, _, val_avg_F1, _, _ = self.evaluate(mode, validation_set,"val")
            #     if lr_schedule_active:
            #         lr_scheduler.step(val_avg_F1)

    def train(self, optimizer, dataset, mode='binary'):
        if mode not in ['binary', 'multi']:
            raise ValueError("Mode should be either 'binary' or 'multi'")

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

        if mode == 'multi':
            batch_mature, batch_gory, batch_slapstick, batch_sarcasm = [], [], [], []
            # log_vars = nn.Parameter(torch.zeros((4)))
        
        for index, i in enumerate(sh_train_set):
            mid = sh_train_set[i]['IMDBid']
            
            if mode == 'multi' and sh_train_set[i]['label'] == 0:
                continue

            file_path = "path_to_I3D_features/" + mid + "_rgb.npy"
            if not os.path.isfile(file_path): 
                print(index, "  - mid:", mid)
                continue

            path = "path_to_I3D_features/"
            a1 = np.load(path + mid + "_rgb.npy")
            a2 = np.load(path + mid + "_flow.npy")
            a = a1 + a2
            masked_img = Utils.mask_vector(36, a)
            a = Utils.pad_segment(a, 36, 0)
            image_vec = a

            path = "path_to_VGGish_features/"
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

            if mode == 'binary':
                batch_y.append(sh_train_set[i]['y'])
            elif mode == 'multi':
                batch_mature.append(sh_train_set[i]['mature'])
                batch_gory.append(sh_train_set[i]['gory'])
                batch_slapstick.append(sh_train_set[i]['slapstick'])
                batch_sarcasm.append(sh_train_set[i]['sarcasm'])

            if (len(batch_x) == batch_size or index == len(sh_train_set) - 1) and len(batch_x) > 0:
                optimizer.zero_grad()

                mask = Utils.masking(batch_x)
                batch_x = Utils.pad_features(batch_x)
                batch_x = torch.tensor(batch_x).cuda()
                batch_image = torch.tensor(batch_image).cuda()
                batch_mask_img = torch.tensor(batch_mask_img).cuda()
                batch_audio = torch.tensor(batch_audio).cuda()
                batch_mask_audio = torch.tensor(batch_mask_audio).cuda()
                
                out, mid = self.model(batch_x, 
                                      torch.tensor(mask).cuda(), 
                                      batch_image.float(), 
                                      batch_mask_img, 
                                      batch_audio.float(), 
                                      batch_mask_audio)

                if mode == 'binary':
                    y_pred1 = out.cpu()
                    loss2 = Utils.compute_l2_reg_val(self.task_head) + \
                        F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y))
                    total_loss += loss2.item()
                    loss2.backward()

                elif mode == 'multi':
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
                if mode == 'binary':
                    batch_y = []
                elif mode == 'multi':
                    batch_mature, batch_gory, batch_slapstick, batch_sarcasm = [], [], [], []

    def evaluate_binary(self, json_data, task, batch_size=24, text_pad_length=500, img_pad_length=36, audio_pad_length=63, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
                # batch_pred = batch["binary_label"].to(device) # batch_size by 2
                batch_pred = batch["binary"].to(device) # batch_size by 2

                # batch_size by 2 for binary
                # batch size by 4 by 2 for multi task
                # out = self.model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio, task)

                out, [] = self.model(batch_text, batch_text_mask, batch_image, batch_mask_img, batch_audio, batch_mask_audio)
                
                
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
        
        print(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        
        return avg_loss, accuracy, f1


    def evaluate_multi(self, dataset, division):
        pass
    
    # def evaluate(self, mode, dataset, division):

    def evaluate(self, mode, json_data, division):
        self.set_eval_mode()

        if mode == "binary":
            return self.evaluate_binary(json_data, mode)
        
        return

        # return self.evaluate_multi(dataset, division)

        total_loss1 = 0
        batch_x, batch_image, batch_mask_img = [], [], []
        batch_audio, batch_mask_audio = [], []
        imdb_ids = []
        predictions = [[], [], []]
        y1_true = []
        label_true = []
        count_loop = 0
        batch_size = C.batch_size

        if mode == 'multi':
            batch_mature, batch_gory, batch_slapstick, batch_sarcasm = [], [], [], []
            mature_true, gory_true, sarcasm_true, slapstick_true = [], [], [], []
            predictions_mature, predictions_gory, predictions_slapstick, predictions_sarcasm = [], [], [], []
            predictions_all = []

        with torch.no_grad():

            for index, i in enumerate(dataset):
                mid = dataset[i]['IMDBid']

                if mode == 'multi' and dataset[i]['label'] == 0 and index != len(dataset) - 1:
                    continue
                
                if mode == "multi":
                    import pdb; pdb.set_trace()
                imdb_ids.append(mid)
                batch_x.append(np.array(dataset[i]['indexes']))

                file_path = "path_to_I3D_features/" + mid + "_rgb.npy"
                if not os.path.isfile(file_path): 
                    count_loop += 1
                    continue

                path = "path_to_I3D_features/"
                a1 = np.load(path + mid + "_rgb.npy")
                a2 = np.load(path + mid + "_flow.npy")
                a = a1 + a2
                masked_img = Utils.mask_vector(36, a)
                a = Utils.pad_segment(a, 36, 0)
                image_vec = a
                batch_image.append(image_vec)
                batch_mask_img.append(masked_img)

                path = "path_to_VGGish_features/"
                try:
                    audio_arr = np.load(path + mid + "_vggish.npy")
                except:
                    audio_arr = np.array([128 * [0.0]])
                masked_audio = Utils.mask_vector(63, audio_arr)
                audio_vec = Utils.pad_segment(audio_arr, 63, 0)
                batch_audio.append(audio_vec)
                batch_mask_audio.append(masked_audio)
                
                if mode == 'binary':
                    batch_y1.append(dataset[i]['y'])
                    y1_true.append(C.label_to_idx[dataset[i]['label']])
                
                elif mode == 'multi':
                    batch_mature.append(dataset[i]['mature'])
                    batch_gory.append(dataset[i]['gory'])
                    batch_slapstick.append(dataset[i]['slapstick'])
                    batch_sarcasm.append(dataset[i]['sarcasm'])

                    mature_label_sample = np.argmax(np.array(dataset[i]['mature']))
                    gory_label_sample = np.argmax(np.array(dataset[i]['gory']))
                    sarcasm_label_sample = np.argmax(np.array(dataset[i]['sarcasm']))
                    slapstick_label_sample = np.argmax(np.array(dataset[i]['slapstick']))

                    mature_true.append(mature_label_sample)
                    import pdb; pdb.set_trace()
                    gory_true.append(gory_label_sample)
                    slapstick_true.append(slapstick_label_sample)
                    sarcasm_true.append(sarcasm_label_sample)

                    label_true.append([mature_label_sample, gory_label_sample, slapstick_label_sample, sarcasm_label_sample])

                if (len(batch_x) == batch_size or index == len(dataset) - 1) and len(batch_x) > 0:
                    mask = Utils.masking(batch_x)
                    batch_x = Utils.pad_features(batch_x)
                    batch_x = torch.tensor(batch_x).cuda()
                    
                    batch_image = np.array(batch_image)
                    batch_image = torch.tensor(batch_image).cuda()
                    
                    batch_mask_img = np.array(batch_mask_img)
                    batch_mask_img = torch.tensor(batch_mask_img).cuda()

                    batch_audio = np.array(batch_audio)
                    batch_audio = torch.tensor(batch_audio).cuda()

                    batch_mask_audio = np.array(batch_mask_audio)
                    batch_mask_audio = torch.tensor(batch_mask_audio).cuda()

                    out, mid_level_out = self.model(batch_x, 
                                                    torch.tensor(mask).cuda(), 
                                                    batch_image.float(), 
                                                    batch_mask_img, 
                                                    batch_audio.float(), 
                                                    batch_mask_audio)

                    if mode == 'binary':
                        y_pred1 = out.cpu()
                        predictions[0].extend(list(torch.argmax(y_pred1, -1).numpy()))
                        loss2 = F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y1))
                        total_loss1 += loss2.item()

                    elif mode == 'multi':
                        mature_pred = out[0].cpu()
                        gory_pred = out[1].cpu()
                        slapstick_pred = out[2].cpu()
                        sarcasm_pred = out[3].cpu()

                        pred_mature = torch.argmax(mature_pred, -1).numpy()
                        pred_gory = torch.argmax(gory_pred, -1).numpy()
                        pred_slap = torch.argmax(slapstick_pred, -1).numpy()
                        pred_sarcasm = torch.argmax(sarcasm_pred, -1).numpy()

                        predictions_mature.extend(list(pred_mature))
                        predictions_gory.extend(list(pred_gory))
                        predictions_slapstick.extend(list(pred_slap))
                        predictions_sarcasm.extend(list(pred_sarcasm))

                        loss2 = F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature))
                        total_loss1 += loss2.item()

                    batch_x, batch_image, batch_mask_img, batch_audio, batch_mask_audio = [], [], [], [], []
                    if mode == 'binary':
                        batch_y1 = []
                        if count_loop+batch_size > len(dataset) - 1:
                            break
                    elif mode == 'multi':
                        batch_mature, batch_gory, batch_slapstick, batch_sarcasm = [], [], [], []

                count_loop += 1

        if mode == 'binary':
            s1, s2, t1, t2 = 0, 0, 0, 0
            for idx, j in enumerate(y1_true):
                if j == 0:
                    t1 += 1
                    if predictions[0][idx] == 0:
                        s1 += 1
                else:
                    t2 += 1
                    if predictions[0][idx] == 1:
                        s2 += 1
            if t1 != 0 and t2 != 0:
                metric = ((s1 / t1) + (s2 / t2)) / 2.0
            else:
                metric = 0

            import pdb; pdb.set_trace()

            print (confusion_matrix(y1_true, predictions[0]))
            print('weighted', f1_score(y1_true, predictions[0], average='weighted'))
            print('micro', f1_score(y1_true, predictions[0], average='micro'))
            print ("****************")
            print('macro', f1_score(y1_true, predictions[0], average='macro'))
            print('f1-score', f1_score(y1_true, predictions[0]))

            return predictions, \
                total_loss1 / len(dataset), \
                    metric, \
                    f1_score(y1_true, predictions[0], average='micro'), \
                    f1_score(y1_true, predictions[0], average='macro'), \
                    f1_score(y1_true, predictions[0], average='weighted'), \
                    f1_score(y1_true, predictions[0])
        
        elif mode == 'multi':
            true_values, preds = [], []
            for i in range(len(mature_true)):
                true_values.append([mature_true[i], gory_true[i], slapstick_true[i], sarcasm_true[i]])
                preds.append([predictions_mature[i], predictions_gory[i], predictions_slapstick[i], predictions_sarcasm[i]])

            print ("acc_score ",accuracy_score(true_values,preds))
            print ("Hamin_score", Utils.hamming_score(np.array(true_values),np.array( preds)))
            print("Hamming_loss:", hamming_loss(true_values, preds))
            print(hamming_loss(true_values, preds) + Utils.hamming_score(np.array(true_values),np.array( preds)))

            print(classification_report(true_values,preds))
            F1_score_mature = f1_score(mature_true, predictions_mature)
            F1_score_gory = f1_score(gory_true, predictions_gory)
            F1_score_slap = f1_score(slapstick_true, predictions_slapstick)
            F1_score_sarcasm = f1_score(sarcasm_true, predictions_sarcasm)
            
            Average_F1_score = (F1_score_mature + F1_score_gory + F1_score_slap + F1_score_sarcasm)/4
            print ("Average_F1_score:", Average_F1_score)

            label_true = np.array(label_true)
            predictions_all = np.array(predictions_all)
            
            print('macro All:', f1_score(label_true, predictions_all, average='macro'))
            
            learning_rate = 1.5e-5
            run_name = 'Test_Multitask_Test_'+str(mature_w)+'_'+str(gory_w)+'_'+str(slap_w)+'_'+str(sarcasm_w)+'_'+str(learning_rate)

            output_dir_path = './'+ run_name + '/'
            if not os.path.exists(output_dir_path):
                os.mkdir(output_dir_path)
            
            path_res_out = os.path.join(output_dir_path, 'res_'+run_name+'.out')
            f = open(path_res_out, "a")
            
            f.write('macro All: %f\n' % f1_score(label_true, predictions_all, average='macro'))
            
            print ("Confusion Matrix All:")
            confusion_matrix_all = multilabel_confusion_matrix(label_true, predictions_all)
            print (multilabel_confusion_matrix(label_true, predictions_all))
            
            print ("Mature")
            print (confusion_matrix(mature_true, predictions_mature))
            print('weighted', f1_score(mature_true, predictions_mature, average='weighted'))
            print('micro', f1_score(mature_true, predictions_mature, average='micro'))
            print('macro', f1_score(mature_true, predictions_mature, average='macro'))
            print('None', f1_score(mature_true, predictions_mature, average=None))
            print ("============================")
            
            f.write ("Mature\n")
            f.write('weighted: %f\n' % f1_score(mature_true, predictions_mature, average='weighted'))
            f.write('micro: %f\n' % f1_score(mature_true, predictions_mature, average='micro'))
            f.write('macro: %f\n' % f1_score(mature_true, predictions_mature, average='macro'))
            f.write ("============================\n")
            
            print ("Gory")
            print (confusion_matrix(gory_true, predictions_gory))
            print('weighted', f1_score(gory_true, predictions_gory, average='weighted'))
            print('micro', f1_score(gory_true, predictions_gory, average='micro'))
            print('macro', f1_score(gory_true, predictions_gory, average='macro'))
            print('None', f1_score(gory_true, predictions_gory, average=None))
            print ("=============================")

            f.write ("Gory\n")
            f.write('weighted: %f\n' % f1_score(gory_true, predictions_gory, average='weighted'))
            f.write('micro: %f\n' % f1_score(gory_true, predictions_gory, average='micro'))
            f.write('macro: %f\n' % f1_score(gory_true, predictions_gory, average='macro'))
            f.write('binary: %f\n' % f1_score(gory_true, predictions_gory, average='binary'))
            f.write ("============================\n")
            
            print ("Slapstick")
            print (confusion_matrix(slapstick_true, predictions_slapstick))
            print('weighted', f1_score(slapstick_true, predictions_slapstick, average='weighted'))
            print('micro', f1_score(slapstick_true, predictions_slapstick, average='micro'))
            print('macro', f1_score(slapstick_true, predictions_slapstick, average='macro'))
            print('None', f1_score(slapstick_true, predictions_slapstick, average=None))
            print ("=============================")

            f.write ("Slapstick\n")
            f.write('weighted: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='weighted'))
            f.write('micro: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='micro'))
            f.write('macro: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='macro'))
            f.write('binary: %f\n' % f1_score(slapstick_true, predictions_slapstick, average='binary'))
            f.write ("============================\n")
        
            print ("Sarcasm")
            print (confusion_matrix(sarcasm_true, predictions_sarcasm))
            print('weighted', f1_score(sarcasm_true, predictions_sarcasm, average='weighted'))
            print('micro', f1_score(sarcasm_true, predictions_sarcasm, average='micro'))
            print('macro', f1_score(sarcasm_true, predictions_sarcasm, average='macro'))
            print('None', f1_score(sarcasm_true, predictions_sarcasm, average=None))
            print ("=============================")

            f.write ("Sarcasm\n")
            f.write('weighted: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='weighted'))
            f.write('micro: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='micro'))
            f.write('macro: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='macro'))
            f.write('binary: %f\n' % f1_score(sarcasm_true, predictions_sarcasm, average='binary'))
            f.write ("============================\n")
        
            f.write('acc_score: %f\n' % accuracy_score(true_values,preds))
            f.write('Hamin_score: %f\n'% Utils.hamming_score(np.array(true_values),np.array( preds)))
            f.write('Hamming_loss: %f\n'% hamming_loss(true_values, preds))
            f.close()

            return predictions, \
                total_loss1 / len(dataset), \
                F1_score_mature, \
                F1_score_gory, \
                F1_score_slap, \
                F1_score_sarcasm, \
                Average_F1_score, \
                confusion_matrix_all, label_true, predictions_all
            
    def test(self):
        model = self.task_head
    
        test_pred, test_loss1, test_F1_mature, test_F1_gory, test_F1_slap, test_F1_sarcasm, test_avg_F1, confusion_matrix_all, label_true, predictions_all = evaluate(model, test_set, "test")

        print('Test Loss %.5f, Test F1 Mature %.5f, Test F1 Gory %.5f, Test F1 Slap %.5f, Test F1 Sarcasm %.5f, Test F1 Average %.5f' % (test_loss1, test_F1_mature, test_F1_gory, test_F1_slap, test_F1_sarcasm, test_avg_F1))

