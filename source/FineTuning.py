

class Naive:
    def __init__(self, heads):
        pass

    def start_iter(self):
        pass

    def end_iter(self):
        pass

    def backward(self, outputs):
        for loss in outputs.values():
            assert(loss.requires_grad == True)
            loss.backward(retain_graph=True)

    def process_eval(self, loss, accuracy):
        pass

    def get_eval_iter_count(self):
        return float('inf')

class Weighted:
    def __init__(self, heads):
        self.weights = {
            'mature': 0.1,
            'gory': 0.4,
            'slapstick': 0.2,
            'sarcasm': 0.2
        }

    def start_iter(self):
        pass

    def end_iter(self):
        pass

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

    def process_eval(self, loss, accuracy):
        # Adjust the weights based on loss and/or accuracy
        pass

    def get_eval_iter_count(self):
        return float('inf')
            
class DSGAttributes:
    def __init__(self):
        self.mode = "go"
        self.last_loss = [None, None]
        self.best_loss = None
        self.stop_idx = None

class DynamicStopAndGo:
    def __init__(self, heads):
        self.loss_threshold = 0.1
        self.accuracy_threshold = 0.5
        self.iter_gap = 3
        self.iter_count = 0

        self.dsg_data = {}
        for head in heads:
            self.dsg_data[head] = DSGAttributes()

    def start_iter(self):
        self.iter_count += 1

    def end_iter(self):
        pass

    def percentage(self, current, previous):
        if previous == 0:
            return 0
        return (abs(current - previous) / previous) * 100.0

    def backward(self, outputs):
        for head, attr in self.dsg_data.items():
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
        for head, attr in self.dsg_data.items():
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
                attr.mode = "stop"
                # import pdb; pdb.set_trace()

                attr.stop_idx = self.iter_count
                self.update_losses(attr, current_loss)
            else:
                if current_loss > attr.best_loss and \
                    self.percentage(current_loss, attr.best_loss) >= 0.5:
                    attr.mode = "go"
                    # import pdb; pdb.set_trace()

                    self.update_losses(attr, current_loss)

    def get_eval_iter_count(self):
        return 9
