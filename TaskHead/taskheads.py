import Binary.source.nlp_comic_binary as binary
import Binary.source.models.unified_model_binary as binary_model
import MultiTask.source.nlp_comic_multi_task as multi
import MultiTask.source.models.multi_task_model as multi_model
from torch import optim
import sys

class TaskHead:
    def __init__(self, model, train, eval):
        self.model = model
        self.train = train
        self.eval = eval

class BinaryTaskHead(TaskHead):
    def __init__(self, model, train, eval):
        if binary.optimizer_type == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=binary.learning_rate)
        elif binary.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=binary.learning_rate, 
                                       momentum=0.9, weight_decay=binary.weight_decay_val)
        super(BinaryTaskHead, self).__init__(model, train, eval)       

    def do_train(self):
        self.train(self.model, self.optimizer)

class MultiTaskHead(TaskHead):
    def __init__(self, model, train, eval):
        if multi.optimizer_type == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=multi.learning_rate)
        elif multi.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=multi.learning_rate, 
                                       momentum=0.9, weight_decay=binary.weight_decay_val)
        super(MultiTaskHead, self).__init__(model, train, eval)       

    def do_train(self):
        self.train(self.model, self.optimizer)

class TaskHeads:
    def __init__(self):
        self.heads = {}

    def add(self, name, taskHead):
        self.heads[name] = taskHead

    def train(self, name):
        taskHead = self.heads[name]
        taskHead.do_train()

def setupHeads():
    taskHeads = TaskHeads()
    binTaskHead = BinaryTaskHead(binary_model.Bert_Model(),
                        binary.train, binary.evaluate)
    taskHeads.add("binary", binTaskHead)

    multiTaskHead = MultiTaskHead(multi_model.Bert_Model(), 
                                multi.train, multi.evaluate)
    taskHeads.add("multi", multiTaskHead)
    return taskHeads

def trainHeads(taskHeads):
    for name in taskHeads.heads:
        taskHeads.train(name)

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print("Usage: {} <eval|train>".format(sys.argv[0]))
        sys.exit(1)
    
    mode = sys.argv[2]
    if mode != "eval" and mode != "train":
        print("Usage: {} <eval|train>".format(sys.argv[0]))
        sys.exit(2)

    if mode == "train":
        heads = setupHeads()
        trainHeads(heads)



