import Binary.source.nlp_comic_binary as binary
import Binary.source.models.unified_model_binary as binary_model
from torch import optim

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

class TaskHeads:
    def __init__(self):
        self.TaskHeads = {}

    def add(self, name, TaskHead):
        self.TaskHeads[name] = TaskHead

    def train(self, name):
        TaskHead = self.TaskHeads[name]
        TaskHead.do_train()

TaskHeads = TaskHeads()
bin_TaskHead = BinaryTaskHead(binary_model.Bert_Model(),
                       binary.train, binary.evaluate)
TaskHeads.add("binary", bin_TaskHead)
TaskHeads.train("binary")