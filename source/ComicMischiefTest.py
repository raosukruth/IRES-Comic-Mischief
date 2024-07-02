from ComicMischiefDetection import ComicMischiefDetection, create_encoding_hca
import sys
from HCA import HCA
from FeatureEncoding import FeatureEncoding
import json
import config as C

def run(pretrain):
    feature_encoding, hca = create_encoding_hca()
    
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

    if pretrain:
        ### Do Pretraining here ###
        pass
    
    model_binary = ComicMischiefDetection(head="binary", encoding=feature_encoding, hca=hca)
    model_binary.training_loop(0, 1, train_set, validation_set="train_features_lrec_camera.json")

    model_multi = ComicMischiefDetection(head="multi", encoding=feature_encoding, hca=hca)
    # model_multi.training_loop(0, 1, train_set, validation_set="train_features_lrec_camera.json")
    
    model_binary.test()
    # model_multi.test()

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