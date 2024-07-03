from ComicMischiefDetection import ComicMischiefDetection, create_encoding_hca
import sys

def run(pretrain):
    feature_encoding, hca = create_encoding_hca()
    if pretrain:
        ### Do Pretraining here ###
        model_pretrain = ComicMischiefDetection(heads=["pretrain"], 
                                                encoding=feature_encoding, 
                                                hca=hca)
        model_pretrain.training_loop(0, 1, "train_features_lrec_camera.json", 
                                     "val_features_lrec_camera.json", 
                                     pretrain=True)

    model_train = ComicMischiefDetection(heads=["binary", "multi"], 
                                         encoding=feature_encoding, 
                                         hca=hca)
    model_train.training_loop(0, 1, "train_features_lrec_camera.json", 
                              "val_features_lrec_camera.json")
    model_train.test()

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 2:
        print("Usage: {} [pretrain]".format(sys.argv[0]))
        sys.exit(1)

    if len(args) == 2 and sys.argv[1] != "pretrain":
        print("Usage: {} [pretrain]".format(sys.argv[0]))
        sys.exit(2)

    pretrain = False
    if len(args) == 2:
        pretrain = True
    run(pretrain)