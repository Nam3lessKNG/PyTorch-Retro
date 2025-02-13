import numpy as np
import random
from collections import deque
from PIL import Image
import torch

def downscale_image(image, target_size=(84, 84)):
    image = Image.fromarray(image).convert('L')
    image = image.resize(target_size, Image.Resampling.LANCZOS) 
    return np.array(image, dtype=np.uint8)

def update_frame_stack(stack, new_frame):
    new_stack = torch.zeros_like(stack)
    new_stack[0, :, :] = new_frame  
    new_stack[1:, :, :] = stack[:-1, :, :]  
    return new_stack

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.data = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def insert_data(self, state, action, reward, next_state, terminated, truncated):
        self.data.append((state, action, reward, next_state, terminated, truncated))

    def sample_data(self, num_of_data):
        return random.sample(self.data, min(len(self.data), num_of_data))
    
    def get_length(self):
        return len(self.data)
