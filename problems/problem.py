import abc


class Problem(abc.ABC):
    def __init__(self, 
                 seq_len, # Sequence length
                 buffer_size, # Memory buffer size
                 num_variations=1, # Training variations
                 use_current_obs=False, # Provide current observation to the proxy
                 randomize_reset=True, 
                 observation_shape=None, 
                 observation_type=int, 
                 action_dim=None):
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.num_variations = num_variations
        self.use_current_obs = use_current_obs
        self.randomize_reset = randomize_reset
        self.observation_shape = observation_shape
        self.observation_type = observation_type
        self.action_dim = action_dim
        self.samples = None
        self.sidx = None
    
    @abc.abstractmethod
    def get_sample(self, idx):
        # this function should update the target for the proxy
        pass

    @abc.abstractmethod
    def get_target(self, t):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def reset_sidx(self, sidx):
        pass


class Evaluator(abc.ABC):
    def __init__(self, checkpoint_folder=None):
        self.checkpoint_folder = checkpoint_folder

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        pass
