
class Finetuning:
    def __init__(self):
        pass

    def start_epoch(self):
        pass

    def end_epoch(self):
        pass

    def start_iter(self):
        pass

    def end_iter(self):
        pass

    def backward(self, outputs):
        assert(False, "Must implement in the derived class")

    def process_eval(self, loss, accuracy):
        assert(False, "Shouldn't have been called")
    
    def get_eval_iter_count(self):
        return float('inf')

class Naive(Finetuning):
    def __init__(self, heads, ce=False):
        super(Naive, self).__init__()

    def backward(self, outputs):
        for loss in outputs.values():
            assert(loss.requires_grad == True)
            loss.backward(retain_graph=True)

class Weighted(Finetuning):
    def __init__(self, heads, ce=False):
        super(Weighted, self).__init__()
        self.weights = {
            'mature': 0.1,
            'gory': 0.4,
            'slapstick': 0.2,
            'sarcasm': 0.2
        }

    def backward(self, outputs):
        total_loss = 0
        for head, loss in outputs.items():
            if head not in self.weights:
                assert(loss.requires_grad == True)
                loss.backward(retain_graph=True)
            else:
                total_loss += loss * self.weights[head] 
        if total_loss:
            assert(total_loss.requires_grad == True)
            total_loss.backward(retain_graph=True)
            
class DSGAttributes:
    def __init__(self):
        self.mode = "go"
        self.last_loss = [None, None]
        self.best_loss = None
        self.stop_idx = None
        self.task_state = None

class DynamicStopAndGo(Finetuning):
    def __init__(self, heads, ce=False):
        super(DynamicStopAndGo, self).__init__()
        self.loss_threshold = 0.1
        self.accuracy_threshold = 0.5
        self.iter_gap = 8 # delta
        self.iter_count = 0
        if ce == False:
            self.phases = {
                'phase': {
                    'start': 0,
                    'end': 30,
                    'heads': heads
                }
            }
        else:
            self.phases = \
                {
                    'phase1': {
                        'start': 0,
                        'end': 10,
                        'heads': ["binary"]
                    },
                    'phase2': {
                        'start': 11,
                        'end': 20,
                        'heads': ['mature', 'gory']
                    },
                    'phase3': {
                        'start': 21,
                        'end': 30,
                        'heads': ['slapstick', 'sarcasm']
                    }
                }
        self.dsg_data = {}
        for head in heads:
            self.dsg_data[head] = DSGAttributes()
        self.epoch_num = 0
        
    def start_iter(self):
        self.iter_count += 1

    def percentage(self, current, previous):
        if previous == 0:
            return 0
        return (abs(current - previous) / previous) * 100.0
    
    def get_heads(self):
        for data in self.phases.values():
            if self.epoch_num >= data["start"] and self.epoch_num <= data["end"]:
                return data["heads"]
        return None

    def backward(self, outputs):
        heads = self.get_heads()
        for head in heads:
            attr = self.dsg_data[head]
            run_backwards = False
            if attr.mode == "go":
                run_backwards = True
            else:
                if (self.iter_count - attr.stop_idx) % self.iter_gap == 0:
                    run_backwards = True
            if run_backwards:
                loss = outputs[head]
                assert(loss.requires_grad == True)
                loss.backward(retain_graph=True)                
            
    def update_losses(self, attr, current_loss):
        if not attr.best_loss:
            attr.best_loss = current_loss
        elif current_loss < attr.best_loss:
            attr.best_loss = current_loss
        if attr.mode == "stop":
            attr.last_loss[0] = None
            attr.last_loss[1] = None
            return 
        if attr.last_loss[0] == None:
            attr.last_loss[0] = current_loss
            return
        attr.last_loss[1] = attr.last_loss[0]
        attr.last_loss[0] = current_loss

    def process_eval(self, loss, accuracy):
        heads = self.get_heads()
        for head in heads:
            attr = self.dsg_data[head]
            current_loss = loss[head]
            if attr.mode == "go":
                if attr.last_loss[0] == None:
                    self.update_losses(attr, current_loss)
                    continue
                if attr.last_loss[1] == None:
                    self.update_losses(attr, current_loss)
                    continue
                last_loss = min(attr.last_loss)
                if current_loss > last_loss and \
                    self.percentage(current_loss, last_loss) > 0.1:
                    self.update_losses(attr, current_loss)
                    continue
                attr.task_state = "converged"
                attr.mode = "stop"
                attr.stop_idx = self.iter_count
                self.update_losses(attr, current_loss)
            else:
                if current_loss > attr.best_loss and \
                    self.percentage(current_loss, attr.best_loss) >= 0.5:
                    attr.task_state = "diverged"
                    self.update_losses(attr, current_loss)

    def get_eval_iter_count(self):
        return 16  # nt

class RoundRobin(Finetuning):
    def __init__(self, heads, ce=False):
        super(RoundRobin, self).__init__()
        self.iter_count = 0
        self.heads = heads

    def end_iter(self):
        self.iter_count += 1

    def backward(self, outputs):
        head_index = self.iter_count % len(self.heads)
        head = self.heads[head_index]
        loss = outputs[head]
        assert(loss.requires_grad == True)
        loss.backward(retain_graph=True)

class CurriculumLearning(Finetuning):
    def __init__(self, heads, ce=True):
        super(CurriculumLearning, self).__init__()
        self.heads = heads
        self.epoch_num = 0
        self.phases = \
            {
                'phase1': {
                    'start': 0,
                    'end': 10,
                    'heads': ["binary"]
                },
                'phase2': {
                    'start': 11,
                    'end': 20,
                    'heads': ['mature', 'gory']
                },
                'phase3': {
                    'start': 21,
                    'end': 30,
                    'heads': ['slapstick', 'sarcasm']
                }
            }

    def start_epoch(self):
        self.epoch_num += 1

    def get_heads(self):
        for data in self.phases.values():
            if self.epoch_num >= data["start"] and self.epoch_num <= data["end"]:
                return data["heads"]
        return None

    def backward(self, outputs):
        heads = self.get_heads()
        for head in heads:
            loss = outputs[head]
            assert(loss.requires_grad == True)
            loss.backward(retain_graph=True)