import os
import numpy as np


class ReplayBuffer():
    def __init__(self,max_size, input_shape, n_actions, seed=None):
        self.mem_size = int(max_size)
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.rng = np.random.default_rng(seed)

    def store_transition(self, state, action, reward, next_state,done):
        index= self.mem_cntr% self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr +=1 
    

    def sample_buffer(self, batch_size):
        max_mem= min(self.mem_size, self.mem_cntr)
        batch = self.rng.integers(0, max_mem, batch_size)

        state = self.state_memory[batch]
        next_state = self.next_state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        done = self.terminal_memory[batch]


        return state, action, reward, next_state, done

    def save(self, path):
        n = min(self.mem_cntr, self.mem_size)
        tmp = path + ".tmp.npz"
        np.savez(tmp, state=self.state_memory[:n], next_state=self.next_state_memory[:n],
                 action=self.action_memory[:n], reward=self.reward_memory[:n],
                 terminal=self.terminal_memory[:n], mem_cntr=self.mem_cntr)
        os.replace(tmp, path)

    def load(self, path):
        d = np.load(path)
        n = len(d["reward"])
        self.state_memory[:n] = d["state"]
        self.next_state_memory[:n] = d["next_state"]
        self.action_memory[:n] = d["action"]
        self.reward_memory[:n] = d["reward"]
        self.terminal_memory[:n] = d["terminal"]
        self.mem_cntr = int(d["mem_cntr"])
