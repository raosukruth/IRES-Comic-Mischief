import os
import sys

import random
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(7)
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False

from torch import nn
import time
import json
import numpy as np
import pandas as pd
from random import shuffle
from torch import optim
from torch.nn import functional as F
from models.multi_task_model import *
import config as C
from sklearn.metrics import f1_score
import warnings
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from pytorch_pretrained_bert import BertAdam
from transformers import AdamW
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

torch_helper = TorchHelper()


loss_weights1 = torch.Tensor([1,3])


run_mode = 'run'
criterian = nn.CrossEntropyLoss()


start_epoch = 0
batch_size = C.batch_size

max_epochs = 50
learning_rate = 1.5e-5
clip_grad = 0.5
weight_decay_val = 0
optimizer_type = 'adam'  # sgd


collect_attention = True
run_multitask = False

l2_regularize = True
l2_lambda = 0.1

# Learning rate scheduler
lr_schedule_active = False
reduce_on_plateau_lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau

mature_w = 0.1
gory_w = 0.4
slap_w = 0.2
sarcasm_w = 0.2
run_name = 'Test_Multitask_Test_'+str(mature_w)+'_'+str(gory_w)+'_'+str(slap_w)+'_'+str(sarcasm_w)+'_'+str(learning_rate)
description = ''

output_dir_path = './'+ run_name + '/'
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)

path_res_out = os.path.join(output_dir_path, 'res_'+run_name+'.out')
f = open(path_res_out, "a")
f.write('-------------------------\n')
f.close()

# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------

features_dict_train = json.load(open(C.training_features))
features_dict_val = json.load(open(C.val_features))
features_dict_test = json.load(open(C.test_features))


train_set = features_dict_train
print (len(train_set))
print('Train Loaded')

validation_set = features_dict_val
print (len(validation_set))
print('Validation Loaded')

test_set = features_dict_test
print (len(test_set))
print('test Loaded')

# ------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
def create_model():

    model =  Bert_Model()
    model.cuda()
    return model


def compute_l2_reg_val(model):
    if not l2_regularize:
        return 0.

    l2_reg = None

    for w in model.parameters():
        if l2_reg is None:
            l2_reg = w.norm(2)
        else:
            l2_reg = l2_reg + w.norm(2)

    return l2_lambda * l2_reg.item()



def masking(docs_ints, seq_length=500):

    masks = np.zeros((len(docs_ints), seq_length), dtype=int)

    for i, row in enumerate(docs_ints):
        #mask[i, :len(row)] = 1
        masks[i, -len(row):] = 1

    return masks

def mask_vector(max_size,arr):
   
    if arr.shape[0] > max_size:
       output = [1]*max_size
    else:
       len_zero_value = max_size -  arr.shape[0]
       output = [1]*arr.shape[0] + [0]*len_zero_value
    
    return np.array(output)

def pad_segment(feature, max_feature_len, pad_idx):
    S, D = feature.shape
    #print (S, D)
    try:
       pad_l =  max_feature_len - S
       # pad
       pad_segment = np.zeros((pad_l, D))
       feature = np.concatenate((feature,pad_segment), axis=0)
       #print (feature.shape)
    except:
       feature = feature[0:max_feature_len]
       #print (feature.shape)
    return feature

def pad_features(docs_ints, seq_length=500):

    features = np.zeros((len(docs_ints), seq_length), dtype=int)

    for i, row in enumerate(docs_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features
# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, optimizer):
    global mature_w, gory_w, slap_w, sarcasm_w

    start_time = time.time()

    model.train()

    batch_idx = 1
    total_loss = 0
    batch_x = []
    batch_image, batch_mask_img = [],[]
    batch_audio, batch_mask_audio = [],[]
    batch_emo_deepMoji = []
    batch_mask = []
    batch_y, batch_mature, batch_gory, batch_slapstick, batch_sarcasm = [], [], [], [],[]
    batch_text = []
    train_imdb = []
    sh_train_set = train_set

    log_vars = nn.Parameter(torch.zeros((4)))
    num_data = 0
    for index, i in enumerate(sh_train_set):
        #list(np.int_(batch_x))
        mid = sh_train_set[i]['IMDBid']
        if sh_train_set[i]['label'] == 0:
            continue

        file_path = "path_to_I3D_features/"+mid+"_rgb.npy"
        if not os.path.isfile(file_path): 
            print(index, "  - mid:", mid)
            continue
        
        path = "path_to_I3D_features/"
        #image_vec = np.load("./deepMoji_out/"+mid+".npy")
        a1 = np.load(path+mid+"_rgb.npy")
        a2 = np.load(path+mid+"_flow.npy")
        a = a1+a2
        masked_img = mask_vector(36,a)
        a = pad_segment(a, 36, 0)
        image_vec = a
        #masked_img = mask_vector(36,a)

        path = "path_to_VGGish_features/"
        try:
           audio_arr = np.load(path+mid+"_vggish.npy")
        except:
           audio_arr = np.array([128*[0.0]])
        masked_audio = mask_vector(63,audio_arr)
    
        audio_vec = pad_segment(audio_arr, 63, 0)
        batch_audio.append(audio_vec)
        batch_mask_audio.append(masked_audio)

        train_imdb.append(mid)
        batch_x.append(np.array(sh_train_set[i]['indexes']))
        batch_mask_img.append(masked_img)
        batch_image.append(image_vec)
        batch_mature.append(sh_train_set[i]['mature'])
        batch_gory.append(sh_train_set[i]['gory'])
        batch_slapstick.append(sh_train_set[i]['slapstick'])
        batch_sarcasm.append(sh_train_set[i]['sarcasm'])

        if (len(batch_x) == batch_size) and len(batch_x)>0:

            optimizer.zero_grad()

            mask = masking(batch_x)
            batch_x = pad_features(batch_x)
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


            out, mid = model(batch_x, torch.tensor(mask).cuda(),batch_image.float(),batch_mask_img, batch_audio.float(),batch_mask_audio)


            mature_pred = out[0].cpu()
            gory_pred = out[1].cpu()
            slapstick_pred = out[2].cpu()
            sarcasm_pred = out[3].cpu()
            loss1 = F.binary_cross_entropy(mature_pred, torch.Tensor(batch_mature)) 
            loss2 = F.binary_cross_entropy(gory_pred, torch.Tensor(batch_gory))
            loss3 = F.binary_cross_entropy(slapstick_pred, torch.Tensor(batch_slapstick))
            loss4 = F.binary_cross_entropy(sarcasm_pred, torch.Tensor(batch_sarcasm))
            total_loss += loss1.item() + loss2.item() + loss3.item() + loss4.item()
            
            loss = mature_w*loss1 + gory_w*loss2 + slap_w*loss3 + sarcasm_w*loss4
            loss.backward()


            optimizer.step()

            torch_helper.show_progress(batch_idx, np.ceil(len(sh_train_set) / batch_size), start_time,
                                       round(total_loss / (index + 1), 4))
            batch_idx += 1
            batch_x, batch_y,batch_image,batch_mask_img = [], [], [],[]
            batch_audio, batch_mask_audio = [],[]
            batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []
            #break

    return model

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        c = 0
        for j in range(len(y_true[i])):
            if y_true[i][j] == y_pred[i][j]:
               c += 1
        acc_list.append (c/4)
    return np.mean(acc_list)


# ----------------------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------------------
def evaluate(model, dataset, division):
    model.eval()

    total_loss = 0
    total_loss1, total_loss2, total_loss3 = 0, 0, 0

    batch_x, batch_y1,batch_image,batch_mask_img = [], [],[],[]
    batch_director = []
    batch_genre = []
    y1_true, y2_true, y3_true = [], [], []
    imdb_ids = []
    predictions = [[], [], []]
    id_to_vec = {}
    batch_audio, batch_mask_audio = [],[]
    #batch_audio, batch_mask_audio = [],[]
    batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []
    mature_true, gory_true, sarcasm_true, slapstick_true = [], [], [], []
    batch_text = []
    predictions_mature, predictions_gory, predictions_slapstick, predictions_sarcasm = [], [], [], []
    predictions_all, label_true = [], []
    with torch.no_grad():
        list_names = []
        for index,i in enumerate(dataset):
            mid = dataset[i]['IMDBid']
            if dataset[i]['label'] == 0 and index != len(dataset) - 1:
                continue
            
            imdb_ids.append(mid)
            batch_x.append(np.array(dataset[i]['indexes']))

            file_path = "path_to_I3D_features/"+mid+"_rgb.npy"
            if not os.path.isfile(file_path): 
                count_loop += 1
                continue
            
            path = "path_to_I3D_features/"
            #image_vec = np.load("./deepMoji_out/"+mid+".npy")
            a1 = np.load(path+mid+"_rgb.npy")
            a2 = np.load(path+mid+"_flow.npy")
            a = a1+a2
            masked_img = mask_vector(36,a)
            a = pad_segment(a, 36, 0)
            image_vec = a
            batch_image.append(image_vec)
            #masked_img = mask_vector(36,a)
            batch_mask_img .append(masked_img)


            path = "path_to_VGGish_features/"
            try:
                audio_arr = np.load(path+mid+"_vggish.npy")
            except:
                audio_arr = np.array([128*[0.0]])

            masked_audio = mask_vector(63,audio_arr)
            #print (masked_audio)
            audio_vec = pad_segment(audio_arr, 63, 0)
            batch_audio.append(audio_vec)
            batch_mask_audio.append(masked_audio)

            batch_mature.append(dataset[i]['mature'])
            batch_gory.append(dataset[i]['gory'])
            batch_slapstick.append(dataset[i]['slapstick'])
            batch_sarcasm.append(dataset[i]['sarcasm'])

            mature_label_sample = np.argmax(np.array(dataset[i]['mature']))
            gory_label_sample = np.argmax(np.array(dataset[i]['gory']))
            sarcasm_label_sample = np.argmax(np.array(dataset[i]['sarcasm']))
            slapstick_label_sample = np.argmax(np.array(dataset[i]['slapstick']))
            
            mature_true.append(mature_label_sample)
            gory_true.append(gory_label_sample)
            slapstick_true.append(slapstick_label_sample)
            sarcasm_true.append(sarcasm_label_sample)

            label_true.append([mature_label_sample,gory_label_sample,slapstick_label_sample,sarcasm_label_sample])

            if (len(batch_x) == batch_size or index == len(dataset) - 1) and len(batch_x)>0:

                mask = masking(batch_x)

                #print (mask)
                batch_x = pad_features(batch_x)
                batch_x = np.array(batch_x)
                batch_x = torch.tensor(batch_x).cuda()

                batch_image = np.array(batch_image)
                batch_image = torch.tensor(batch_image).cuda()

                batch_mask_img = np.array(batch_mask_img )
                batch_mask_img = torch.tensor(batch_mask_img ).cuda()

                batch_audio = np.array(batch_audio)
                batch_audio = torch.tensor(batch_audio).cuda()
 
                batch_mask_audio = np.array(batch_mask_audio)
                batch_mask_audio = torch.tensor(batch_mask_audio).cuda()

                out, mid_level_out = model(batch_x, torch.tensor(mask).cuda(),batch_image.float(),batch_mask_img,batch_audio.float(),batch_mask_audio)

                #mature_pred = out[0].cpu()
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
                # _, labels1 = torch.Tensor(batch_y1).max(dim=1)

                total_loss1 += loss2.item()

                batch_x, batch_y1,batch_image,batch_mask_img  = [], [], [],[]
                batch_director = []
                batch_genre = []
                batch_mask = []
                batch_text = []
                batch_similar = []
                batch_description = []
                imdb_ids = []
                batch_audio, batch_mask_audio = [],[]
                batch_mature, batch_gory, batch_sarcasm, batch_slapstick = [], [], [], []

    true_values = []
    preds = []
    from sklearn.metrics import hamming_loss
    for i in range(len(mature_true)):
         true_values.append([mature_true[i], gory_true[i], slapstick_true[i], sarcasm_true[i]])
         preds.append([predictions_mature[i],predictions_gory[i],predictions_slapstick[i],predictions_sarcasm[i]])
    print ("acc_score ",accuracy_score(true_values,preds))
    print ("Hamin_score",hamming_score(np.array(true_values),np.array( preds)))
    print("Hamming_loss:", hamming_loss(true_values, preds))
    print (hamming_loss(true_values, preds) + hamming_score(np.array(true_values),np.array( preds)))
    from sklearn.metrics import classification_report
    print (classification_report(true_values,preds))
    F1_score_mature = f1_score(mature_true, predictions_mature)
    F1_score_gory = f1_score(gory_true, predictions_gory)
    F1_score_slap = f1_score(slapstick_true, predictions_slapstick)
    F1_score_sarcasm = f1_score(sarcasm_true, predictions_sarcasm)
    
    Average_F1_score = (F1_score_mature + F1_score_gory + F1_score_slap + F1_score_sarcasm)/4
    print ("Average_F1_score:", Average_F1_score)

    label_true = np.array(label_true)
    predictions_all = np.array(predictions_all)
    
    print('macro All:', f1_score(label_true, predictions_all, average='macro'))
    
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
    f.write('Hamin_score: %f\n'% hamming_score(np.array(true_values),np.array( preds)))
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

def training_loop():

    model = create_model()
    
    if optimizer_type == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay_val)
   
    lr_scheduler = reduce_on_plateau_lr_schdlr(optimizer, 'max', min_lr=1e-8, patience=2, factor=0.5)

    max_test_avg_F1 = 0
    import csv
    
    for epoch in range(start_epoch, max_epochs):
        print('[Epoch %d] / %d : %s' % (epoch + 1, max_epochs, run_name))
        f = open(path_res_out, "a")
        f.write('[Epoch %d] / %d : %s\n' % (epoch + 1, max_epochs, run_name))


        model = train(model, optimizer)

        val_pred, val_loss1, val_F1_mature, val_F1_gory, val_F1_slap, val_F1_sarcasm, val_avg_F1, confusion_matrix_all, label_true, predictions_all = evaluate(model, validation_set,"val")
        

        current_lr = 0
        for pg in optimizer.param_groups:
            current_lr = pg['lr']

        if lr_schedule_active:
            lr_scheduler.step(val_avg_F1)

        is_best = torch_helper.checkpoint_model(model, optimizer, output_dir_path, test_avg_F1, epoch + 1,
                                                'max')
        f.close()
        

def test():
    model = create_model()
 
    test_pred, test_loss1, test_F1_mature, test_F1_gory, test_F1_slap, test_F1_sarcasm, test_avg_F1, confusion_matrix_all, label_true, predictions_all = evaluate(model, test_set, "test")

    print('Test Loss %.5f, Test F1 Mature %.5f, Test F1 Gory %.5f, Test F1 Slap %.5f, Test F1 Sarcasm %.5f, Test F1 Average %.5f' % (test_loss1, test_F1_mature, test_F1_gory, test_F1_slap, test_F1_sarcasm, test_avg_F1))

if __name__ == '__main__':
    
    if run_mode != 'test':
        training_loop()
  
    else:
        test()
