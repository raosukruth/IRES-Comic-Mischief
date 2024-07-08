import torch

processed_data_dir = "../processed_data/"

path_to_VGGish_I3D = "/storage/project/Comic_Mischief/i3D-vggish-features/"
path_to_VGGish_features = path_to_VGGish_I3D + "vgg_vecs_extended_merged/"
path_to_I3D_features = path_to_VGGish_I3D + "i3d_vecs_extended_merged/"

supported_heads = ['binary', 'mature', 'gory', 'slapstick', 'sarcasm']

use_gpu = False
if use_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # The following works only on joshi
        # For other systems set it to 0 or find the number of GPUs
        # and decide (Run Cuda.py for help)
        device_id = 5
        torch.cuda.set_device(device_id)
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu") 
else:
    device = torch.device("cpu") # Hack to overcome CUDA out of memory condtion
assert(device)

show_training_loss = True
batch_size = 24
