import random 

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

    def process_eval(self, loss, accuracy, f1_score):
        pass
        
    def backward(self, outputs):
        assert(False, "Must implement in the derived class")

    def process_batch_eval(self, loss, accuracy, f1_score):
        assert(False, "Shouldn't have been called")
    
    def get_batch_eval_iter_count(self):
        return float('inf')

class Discrete(Finetuning):
    def __init__(self, heads):
        super(Discrete, self).__init__()

    def backward(self, outputs):
        for loss in outputs.values():
            assert(loss.requires_grad == True)
            loss.backward(retain_graph=True)

class Weighted(Finetuning):
    def __init__(self, heads):
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
    def __init__(self, heads):
        super(DynamicStopAndGo, self).__init__()
        self.loss_threshold = 0.1
        self.accuracy_threshold = 0.5
        self.iter_gap = 8 # delta
        self.iter_count = 0
        self.phases = {
            'phase': {
                'start': 0,
                'end': 30,
                'heads': heads
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
            if self.epoch_num >= data["start"] and self.epoch_num < data["end"]:
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

    def process_batch_eval(self, loss, accuracy, f1_score):
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

    def get_batch_eval_iter_count(self):
        return 16  # nt

class RoundRobin(Finetuning):
    def __init__(self, heads):
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
    def __init__(self, heads):
        super(CurriculumLearning, self).__init__()
        self.heads = heads
        self.epoch_num = 0
        self.weights = {
            'gory': 0.15,
            'sarcasm': 0.10,
            'slapstick': 0.05,
            'mature': 0.01
        }
        self.curriculum = {
            'step1': {
                'start': 0,
                'end': 5,
                'heads': ['binary']
            },
            'step2': {
                'start': 5,
                'end': 10,
                'heads': ["gory", "binary"]
            },
            'step3': {
                'start': 10,
                'end': 15,
                'heads': ["sarcasm", "gory", "binary"]
            },
            'step4': {
                'start': 15,
                'end': 20,
                'heads': ["slapstick", "sarcasm", "gory", "binary"]
            },
            'step5': {
                'start': 20,
                'end': 35,
                'heads': ["mature", "slapstick", "sarcasm", "gory", "binary"]
            },
        }

    def start_epoch(self):
        self.epoch_num += 1

    def get_heads(self):
        for data in self.curriculum.values():
            if self.epoch_num >= data["start"] and self.epoch_num <= data["end"]:
                return data["heads"]
        return None
    
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

class DynamicWeighted(Finetuning):
    def __init__(self, heads):
        super(DynamicWeighted, self).__init__()

    def combined_loss(self, losses, weights):
        combined_loss = 0
        for i in range(len(losses)):
            weighted_loss = losses[i] * weights[i]
            combined_loss += weighted_loss
        return combined_loss

    def backward(self, outputs):
        losses = []
        for head, loss in outputs.items():
            losses.append(loss)
        if losses:
            inverse_losses = []
            for loss in losses:
                inverse_losses.append(1.0 / loss)
            
            total_inverse_loss = sum(inverse_losses)
            weights = []
            for loss in inverse_losses:
                weights.append(loss / total_inverse_loss)
            
            combined_loss = self.combined_loss(losses, weights)
            assert combined_loss.requires_grad == True
            combined_loss.backward(retain_graph=True)
 
class Assessment(Finetuning):
    def __init__(self, heads, epochs=5, anti=False):
        self.epoch_count = 0
        self.sum_f1_score = {}
        for head in heads:
            self.sum_f1_score[head] = 0
        self.curriculum = {}
        self.assessment_epochs = epochs
        self.heads = heads
        self.anti = anti

    def prepare_curriculum(self):
        sorted_heads = sorted(self.sum_f1_score.items(), key=lambda item: item[1], reverse=not(self.anti))
        step_count = 1
        start = 0
        end = 0
        num_epochs = 5
        previous_heads = []
        for index, (head, _) in enumerate(sorted_heads):
            start = end
            end = start + num_epochs
            if index == (len(sorted_heads) - 1):
                # Give a large number of epochs for the last step
                end += (3 * num_epochs)
            heads = [head]
            heads.extend(previous_heads)
            self.curriculum[f'step{step_count}'] = {
                'start': start, 
                'end': end,
                'heads': heads
            }
            previous_heads = heads
            step_count += 1

    def end_epoch(self):
        self.epoch_count += 1
        # If it is the last epoch, prepare the curriculum and print it
        if self.epoch_count == self.assessment_epochs:
            self.prepare_curriculum()
            print(self.curriculum)

    def process_eval(self, loss, accuracy, f1_score):
        # Keep a sum of f1 scores 
        for head in self.heads:
            self.sum_f1_score[head] += f1_score[head]

    def backward(self, outputs):
        dw = DynamicWeighted(self.heads)
        dw.backward(outputs)

class DynamicCurriculumLearning(Finetuning):
    def __init__(self, heads, anti=False):
        super(DynamicCurriculumLearning, self).__init__()
        # In assessment phase we determine the curriculum
        self.current_phase = "assessment"
        self.assess_epochs = 5
        self.assess = Assessment(heads, self.assess_epochs, anti=anti)
        self.epochs = 0
        self.learning_epoch = 0

    def end_epoch(self):
        if self.current_phase == "assessment":
            self.assess.end_epoch()
        else:
            self.learning_epoch += 1
        self.epochs += 1
        if self.epochs == self.assess_epochs:
            self.current_phase = "learning"

    def process_eval(self, loss, accuracy, f1_score):
        if self.current_phase == "assessment":
            self.assess.process_eval(loss, accuracy, f1_score)

    def get_heads(self):
        assert(len(self.assess.curriculum) > 0)
        for data in self.assess.curriculum.values():
            if self.learning_epoch >= data["start"] and self.learning_epoch < data["end"]:
                return data["heads"]
        return None

    def backward(self, outputs):
        if self.current_phase == "assessment":
            self.assess.backward(outputs)
            return
        heads = self.get_heads()
        dw = DynamicWeighted(heads)
        dw.backward(outputs)

class AdaptiveWeighting(Finetuning):
    def __init__(self, heads, initial_weight=1.0, adjust_rate=0.05):
        super(AdaptiveWeighting, self).__init__()
        self.heads = heads
        self.adjust_rate = adjust_rate
        self.head_weights = {}
        self.previous_losses = {}
        self.current_losses = {}
        for head in heads:
            self.head_weights[head] = initial_weight
            self.previous_losses[head] = 0
            self.current_losses[head] = 0

    def adjust_weights(self):
        for head in self.heads:
            prev_loss = self.previous_losses[head]
            curr_loss = self.current_losses[head]
            if curr_loss > prev_loss:
                # Increase the weight if the current loss is higher than the previous loss
                self.head_weights[head] += self.adjust_rate
            elif curr_loss < prev_loss:
                # Decrease the weight if the current loss is lower than the previous loss
                self.head_weights[head] = max(self.head_weights[head] - self.adjust_rate, 0.0)
            # Update the previous loss to the current loss
            self.previous_losses[head] = curr_loss

    def backward(self, outputs):
        # Update current losses from outputs
        for head in self.heads:
            self.current_losses[head] = outputs[head]

        # Adjust the weights based on the current and previous losses
        self.adjust_weights()
        total_loss = 0
        for head, loss in outputs.items():
            weighted_loss = loss * self.head_weights[head]
            total_loss += weighted_loss
        
        assert(total_loss.requires_grad == True)
        total_loss.backward(retain_graph=True)
        
    def process_eval(self, loss, accuracy, f1_score):
        # Update current losses based on evaluation metrics
        for head in self.heads:
            self.current_losses[head] = loss[head]
        # Adjust weights after processing evaluation
        self.adjust_weights()


class DynamicSampling(Finetuning):
    def __init__(self, heads, initial_prob=0.25, adjust_rate=0.05):
        super(DynamicSampling, self).__init__()
        self.heads = heads
        self.adjust_rate = adjust_rate
        self.head_probs = {}
        self.previous_losses = {}
        self.current_losses = {}
        for head in heads:
            self.head_weights[head] = initial_prob
            self.previous_losses[head] = 0
            self.current_losses[head] = 0

    def adjust_sampling_probs(self):
        total_loss = sum(self.current_losses.values())
        if total_loss == 0:
            return
        for head in self.heads:
            if self.current_losses[head] > self.previous_losses[head]:
                self.head_probs[head] = max(self.head_probs[head] - self.adjust_rate, 0.0)
            else:
                self.head_probs[head] += self.adjust_rate
            total_prob = sum(self.head_probs.values())
            for h in self.heads:
                self.head_probs[h] /= total_prob

    def sample_head(self):
        heads = list(self.heads)
        probs = [self.head_probs[head] for head in heads]
        return random.choices(heads, probs)[0]

    def backward(self, outputs):
        head = self.sample_head()
        loss = outputs[head]
        assert loss.requires_grad == True
        loss.backward(retain_graph=True)

    def process_eval(self, loss, accuracy, f1_score):
        for head in self.heads:
            self.current_losses[head] = loss[head]
        self.adjust_sampling_probs()
        self.previous_losses.update(self.current_losses)
