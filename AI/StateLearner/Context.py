import numpy as np
import random

class Context:
    def __init__(self, probs):# assumes 3 variable observations -
        #obs[0] = reward (binary)
        #obs[1] = cue (with m different levels)
        #obs[2] = context (with n different levels)
        self.probs = probs
        self.m_cues = probs.shape[0]
        self.n_contexts = probs.shape[1]
        return None

    def emit(self, cue):
        this_prob = self.probs[int(cue[0]), int(cue[1])]
        return random.choices([True, False], [this_prob, 1-this_prob])
    
    def get_sequence(self, cues):
        rewards = np.ones((len(cues),1))
        for i, cue in enumerate(cues):
            rewards[i] = self.emit(cue)
        return rewards

    def get_block_sequence(self, t_trials, b_blocks):
        cues = np.ndarray((t_trials * b_blocks, 2))
        context = 0
        for b in range(b_blocks):
            for t in range(t_trials):
                cues[t + b*t_trials,0] = random.choice(range(self.m_cues))
                cues[t + b*t_trials,1] = context
            context += 1
            if context >= self.n_contexts:
                context = 0

        rewards = self.get_sequence(cues)
        obs = np.append(rewards, cues, axis=1)
        return obs

    def get_random_sequence(self, t_trials):
        cues = np.ndarray((t_trials, 2))
        for t in range(t_trials):
            cues[t,0] = random.choice(range(self.m_cues))
            cues[t,1] = random.choice(range(self.n_contexts))
        rewards = self.get_sequence(cues)
        obs = np.append(rewards, cues, axis=1)
        return obs