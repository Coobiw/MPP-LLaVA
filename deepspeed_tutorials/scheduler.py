import math
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, base_lr,eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.current_step = 0

    def step(self):
        self.current_step += 1
        new_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.current_step / self.T_max)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)