from ComicMischiefDetection import ComicMischiefDetection, create_encoding_hca
import sys

def run(pretrain, strategy):
    feature_encoding, hca = create_encoding_hca()
    if pretrain:
        ### Load Pretrained Weights Here ###
        pass

    heads = ['binary', 'mature', 'gory', 'slapstick', 'sarcasm']
    model_train = ComicMischiefDetection(heads=heads, 
                                         encoding=feature_encoding, 
                                         hca=hca, strategy=strategy)
    model_train.training_loop(0, 1, "train_features_lrec_camera.json", 
                              "val_features_lrec_camera.json")
    model_train.test()

if __name__ == "__main__":
    args = sys.argv
    supported_args = ["pretrain", "naive", "weighted", "dsg"]
    if len(args) <= 1 or len(args) > 3:
        print("Usage: {} <naive|weighted|dsg> [pretrain]".format(sys.argv[0]))
        sys.exit(1)
    
    pretrain = False
    strategy = None
    for i in range(1, len(args)):
        arg = args[i]
        if arg not in supported_args:
            print("Usage: {} <naive|weighted|dsg> [pretrain]".format(sys.argv[0]))
            sys.exit(2)
        if arg == "pretrain":
            pretrain = True
        if arg == "naive" or arg == "weighted" or arg == "dsg":
            strategy = arg
    if not strategy:
        print("Usage: {} <naive|weighted|dsg> [pretrain]".format(sys.argv[0]))
        sys.exit(3)
    run(pretrain, strategy)