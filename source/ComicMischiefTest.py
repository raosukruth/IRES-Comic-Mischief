from ComicMischiefDetection import ComicMischiefDetection, create_encoding_hca
import sys
from copy import deepcopy
import config as C
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader

def run(pretrain, strategy):
    feature_encoding, hca = create_encoding_hca()
    heads = ['binary', 'mature', 'gory', 'slapstick', 'sarcasm']
    model_train = ComicMischiefDetection(heads=heads, 
                                         encoding=feature_encoding, 
                                         hca=hca, strategy=strategy, 
                                         pretrain=pretrain)
    model_train.training_loop(0, 30, "train_features_lrec_camera.json", 
                              "val_features_lrec_camera.json")
    model_train.test("test_features_lrec_camera.json")

if __name__ == "__main__":
    args = sys.argv
    supported_args = ["pretrain", "discrete", "weighted", "dsg", "roundrobin", "cl", "dw", "dcl", "dacl", "aw"]
    strategy_args = deepcopy(supported_args)
    strategy_args.remove("pretrain")
    usage_args = "|".join(strategy_args)
    usage_args = "<" + usage_args + ">"
    usage_args += " [pretrain]"
    if len(args) <= 1 or len(args) > 3:
        print(f"Usage: {sys.argv[0]} {usage_args}")
        sys.exit(1)
    
    pretrain = False
    strategy = None
    for i in range(1, len(args)):
        arg = args[i]
        if arg not in supported_args:
            print(f"Usage: {sys.argv[0]} {usage_args}")
            sys.exit(2)
        if arg == "pretrain":
            pretrain = True
        if arg in strategy_args:
            strategy = arg
    if not strategy:
        print(f"Usage: {sys.argv[0]} {usage_args}")
        sys.exit(3)
    run(pretrain, strategy)
