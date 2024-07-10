from ComicMischiefDetection import ComicMischiefDetection, create_encoding_hca
import sys

def run(pretrain, strategy, ce):
    feature_encoding, hca = create_encoding_hca()
    heads = ['binary', 'mature', 'gory', 'slapstick', 'sarcasm']
    model_train = ComicMischiefDetection(heads=heads, 
                                         encoding=feature_encoding, 
                                         hca=hca, strategy=strategy, 
                                         pretrain=pretrain,
                                         ce=ce)
    model_train.training_loop(0, 30, "train_features_lrec_camera.json", 
                              "val_features_lrec_camera.json")
    model_train.test()

if __name__ == "__main__":
    args = sys.argv
    supported_args = ["pretrain", "naive", "weighted", "dsg", "roundrobin", "ce"]
    if len(args) <= 1 or len(args) > 4:
        print("Usage: {} <naive|weighted|dsg|roundrobin> [pretrain] [ce]".format(sys.argv[0]))
        sys.exit(1)
    
    pretrain = False
    strategy = None
    ce = False
    for i in range(1, len(args)):
        arg = args[i]
        if arg not in supported_args:
            print("Usage: {} <naive|weighted|dsg|roundrobin> [pretrain] [ce]".format(sys.argv[0]))
            sys.exit(2)
        if arg == "pretrain":
            pretrain = True
        if arg == "naive" or arg == "weighted" or arg == "dsg" or arg == "roundrobin":
            strategy = arg
        if arg == "ce":
            ce = True
    if not strategy:
        print("Usage: {} <naive|weighted|dsg|roundrobin> [pretrain] [ce]".format(sys.argv[0]))
        sys.exit(3)
    run(pretrain, strategy, ce)