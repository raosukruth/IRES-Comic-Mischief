import os
import sys

import random
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
#import pandas as pd
from random import shuffle
from torch import optim
from torch.nn import functional as F
from experiments import utils as U
from models.unified_model_binary import *
import config as C
from sklearn.metrics import f1_score
import warnings
from sklearn.metrics import confusion_matrix
from pytorch_pretrained_bert import BertAdam
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')
 
loss_weights1 = torch.Tensor([1,3])

run_mode = 'run' 
criterian = nn.CrossEntropyLoss()


start_epoch = 0

batch_size = C.batch_size

max_epochs = 25
learning_rate = 1.9e-5
clip_grad = 0.5
weight_decay_val = 0
optimizer_type = 'adam'  

collect_attention = True
run_multitask = False

l2_regularize = True
l2_lambda = 0.1

lr_schedule_active = False
reduce_on_plateau_lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau

run_name = 'Test_binary_'+str(learning_rate)
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

#===========================================================================================================

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
    # print (arr,arr.shape)
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

    # getting the correct rows x cols shape
    features = np.zeros((len(docs_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(docs_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, optimizer):
    """
    Trains the model using the optimizer for a single epoch.
    :param model: pytorch model
    :param optimizer:
    :return:
    """

    start_time = time.time()

    model.train()

    batch_idx = 1
    total_loss = 0
    batch_x = []
    batch_image, batch_mask_img = [],[]
    batch_audio, batch_mask_audio = [],[]
    batch_emo_deepMoji = []
    batch_mask = []
    batch_y = []
    batch_text = []
    train_imdb = []
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
        #print (masked_audio)
        audio_vec = pad_segment(audio_arr, 63, 0)
        batch_audio.append(audio_vec)
        batch_mask_audio.append(masked_audio)

        train_imdb.append(mid)
        batch_x.append(np.array(sh_train_set[i]['indexes']))
        batch_mask_img.append(masked_img)
        batch_image.append(image_vec)
        batch_y.append(sh_train_set[i]['y'])
    
        if (len(batch_x) == batch_size or index == len(sh_train_set) - 1 ) and len(batch_x)>0:
            #try:
                #print("index:", index)
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


                y_pred1 = out.cpu()
                loss2 = compute_l2_reg_val(model) + F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y))
                total_loss += loss2.item()

                loss2.backward()


                optimizer.step()

                torch_helper.show_progress(batch_idx, np.ceil(len(sh_train_set) / batch_size), start_time,
                                           round(total_loss / (index + 1), 4))
                batch_idx += 1
                batch_x, batch_y,batch_image,batch_mask_img = [], [], [],[]
                batch_audio, batch_mask_audio = [],[]
            

    return model




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
    vecs = []
    batch_text = []
    count_loop = 0
    with torch.no_grad():
        list_names = []
        for i in dataset:
            
            mid = dataset[i]['IMDBid']
            
            file_path = "path_to_I3D_features/"+mid+"_rgb.npy"
            if not os.path.isfile(file_path): 
                count_loop += 1
                continue
            
            words = dataset[i]["words"]
            imdb_ids.append(mid)
            batch_x.append(np.array(dataset[i]['indexes']))
            batch_y1.append(dataset[i]['y'])
            y1_true.append(C.label_to_idx[dataset[i]['label']])
            path = "path_to_I3D_features/"
            a1 = np.load(path+mid+"_rgb.npy")
            a2 = np.load(path+mid+"_flow.npy")
            a = a1+a2
            masked_img = mask_vector(36,a)
            a = pad_segment(a, 36, 0)
            image_vec = a
            batch_image.append(image_vec)
            batch_mask_img .append(masked_img)

            path = "/path_to_VGGish_features/"
            try:
                  audio_arr = np.load(path+mid+"_vggish.npy")
            except:
                  audio_arr = np.array([128*[0.0]])

            
            masked_audio = mask_vector(63,audio_arr)
            audio_vec = pad_segment(audio_arr, 63, 0)
            batch_audio.append(audio_vec)
            batch_mask_audio.append(masked_audio)
            if (len(batch_x) == batch_size or count_loop == len(dataset) - 1) and len(batch_x)>0:
                mask = masking(batch_x)

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
                y_pred1 = out.cpu()

                predictions[0].extend(list(torch.argmax(y_pred1, -1).numpy()))
                  
                pred_temp = torch.argmax(y_pred1, -1).numpy()
                list_names = []
                pred_temp = []
                

                loss2 = F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y1))
                total_loss1 += loss2.item()

                batch_x, batch_y1,batch_image,batch_mask_img  = [], [], [],[]
                batch_director = []
                batch_genre = []
                batch_mask = []
                batch_text = []
                batch_similar = []
                batch_description = []
                #imdb_ids = []
                batch_audio, batch_mask_audio = [],[]
                
                if count_loop+batch_size > len(dataset) - 1:
                    break

            count_loop += 1

    s1,s2,t1,t2 = 0,0,0,0
    for idx, j in enumerate(y1_true):
        if j == 0:
           t1 += 1
           if predictions[0][idx] == 0:
              s1 += 1
        else:
           t2 += 1
           if predictions[0][idx] == 1:
              s2 += 1
    print ("scores ===> ", s1/t1, s2/t2)   
    metric = ((s1/t1) + (s2/t2))/2.0
    micro_f1_2 = f1_score(y1_true, predictions[0], average='weighted')
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


def training_loop():
    model = create_model()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay_val)
   
    lr_scheduler = reduce_on_plateau_lr_schdlr(optimizer, 'max', min_lr=1e-8, patience=2, factor=0.5)

    max_test_binary = 0
    max_test_macro = 0
    for epoch in range(start_epoch, max_epochs):
        print('[Epoch %d] / %d : %s' % (epoch + 1, max_epochs, run_name))

        model = train(model, optimizer)

        val_pred, val_loss1, val_f1, val_micro, val_macro, val_weighted, val_binary = evaluate(model, validation_set, "val")
        
        current_lr = 0
        for pg in optimizer.param_groups:
            current_lr = pg['lr']

        if lr_schedule_active:
            lr_scheduler.step(val_binary)

        is_best = torch_helper.checkpoint_model(model, optimizer, output_dir_path, val_binary, epoch + 1,
                                                'max')


def test():
    model = create_model()
    test_pred, test_loss1, test_f1, test_micro, test_macro, test_weighted, test_binary = evaluate(model, test_set, "test")
    print('test set binary F1 score: %.5f' % (test_binary))

if __name__ == '__main__':
    
    if run_mode != 'test':
        training_loop()
    
    else:
        test()

    