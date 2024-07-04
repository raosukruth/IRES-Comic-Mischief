class Naive:
    def __init__(self, heads):
        pass

    def backward(self, outputs):
        for loss in outputs.values():
            loss.requires_grad_()
            loss.backward()

class Weighted:
    def __init__(self, heads):
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
                loss.requires_grad_()
                loss.backward()
            else:
                total_loss += loss * self.weights[head] 
        if total_loss:
            total_loss.requires_grad_()
            total_loss.backward()
            
class DynamicStopAndGo:
    def __init__(self, heads):
        pass

    def backward(self, outputs):
        pass