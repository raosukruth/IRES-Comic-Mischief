from ComicMischiefDetection import ComicMischiefDetection
import sys

def run(model_name, mode):
    if model_name == "binary" and mode == "train":
        model = ComicMischiefDetection(head=model_name)
        model.training_loop(0, 1)

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print("Usage: {} <binary|multi|pretrain> <eval|train>".format(sys.argv[0]))
        sys.exit(1)
    
    model = sys.argv[1]
    if model != "binary" and model != "multi" and model != "pretrain":
        print("Usage: {} <binary|multi|pretrain> <eval|train>".format(sys.argv[0]))
        sys.exit(2)

    mode = sys.argv[2]
    if mode != "eval" and mode != "train":
        print("Usage: {} <binary|multi|pretrain> <eval|train>".format(sys.argv[0]))
        sys.exit(3)

    run(model, mode)