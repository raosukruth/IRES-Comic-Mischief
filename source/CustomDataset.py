import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from Utils import pad_segment, mask_vector, pad_features
import config as C

class CustomDataset(Dataset):
    def __init__(self, json_data, text_pad_length=500, img_pad_length=36, audio_pad_length=63):
        # self.base_dir = os.path.dirname(os.path.abspath(__file__)) 
        # processed_data_dir = os.path.join(self.base_dir, "processed_data")

        processed_data_dir = C.processed_data_dir
        full_json_data_path = os.path.join(processed_data_dir, json_data)
        # the above may be better to be modified

        self.data = json.load(open(full_json_data_path)) # data dictionary
        self.keys = list(self.data.keys()) # hack for __getitem__

        # define pad lengths for each modality
        # note that the pad length refers to number of tokens
        # so audio input is 128 dimensions for each token, but is padded to 36
        self.text_pad_length = text_pad_length 
        self.img_pad_length = img_pad_length
        self.audio_pad_length = audio_pad_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # here we process all the data accordingly
        key = self.keys[idx] # this is the name of the file
        item = self.data[key]
        
        # FIX THIS
        # NOTE THAT ORIGINAL CODE DOES ERROR HANDLING HERE
        # image_path = os.path.join(self.base_dir, "path_to_I3D_features/")

        image_path = C.path_to_I3D_features

        # ERROR HANDLING BELOW NEEDS UNIFYING

        # Load image features
        image_path_rgb = os.path.join(image_path, f"{key}_rgb.npy")
        image_path_flow = os.path.join(image_path, f"{key}_flow.npy")
        if os.path.isfile(image_path_rgb) and os.path.isfile(image_path_flow):
            a1 = torch.load(image_path_rgb)
            a2 = torch.load(image_path_flow)
            image_vec = a1 + a2
            masked_img = mask_vector(self.img_pad_length, image_vec)
            image_vec = pad_segment(image_vec, self.img_pad_length, 0)
        else:
            # print("Image not found")
            image_vec = torch.zeros((self.img_pad_length, 1024)) 
            masked_img = torch.zeros(self.img_pad_length)

        # Load audio features
        # audio_path = os.path.join(self.base_dir, "path_to_VGGish_features/")

        audio_path = C.path_to_VGGish_features
        try:
            audio_vec = np.load(audio_path)
        except FileNotFoundError:
            # print("Audio Not Found")
            audio_vec = torch.zeros((1, 128))
        masked_audio = mask_vector(self.audio_pad_length, audio_vec)
        audio_vec = pad_segment(audio_vec, self.audio_pad_length, 0)

        # Process text
        text = torch.tensor(item['indexes']) # tokenized text
        mask = mask_vector(self.text_pad_length, text)
        text = pad_features([text], self.text_pad_length)[0]

        binary_label = torch.tensor(item['y']) 
        mature = torch.tensor(item["mature"])
        gory = torch.tensor(item["gory"])
        sarcasm = torch.tensor(item["sarcasm"])
        slapstick = torch.tensor(item["slapstick"])

        return {
            'text': text,
            'text_mask': mask,
            'image': image_vec.float(),
            'image_mask': masked_img,
            'audio': audio_vec.float(),
            'audio_mask': masked_audio,
            'binary': binary_label.float(),
            "mature": mature.float(),
            "gory": gory.float(),
            "sarcasm": sarcasm.float(),
            "slapstick": slapstick.float()
        }


if __name__ == "__main__":
    dataset = CustomDataset("test_features_lrec_camera.json")
    idx = 0
    for item in dataset:
        if idx == 1:
            break
        for key, value in item.items():
            print(key)
            print(value)
            print()
        idx += 1