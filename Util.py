import numpy as np
import random
from collections import deque
from PIL import Image
import torch

def downscale_image(image, target_size=(84, 84)):
    image = Image.fromarray(image)
    image = image.convert('L')
    image = image.resize(target_size, Image.BILINEAR)
    return np.array(image)

def update_frame_stack(stack, new_frame):
    # Assuming new_frame is already on the correct device
    new_stack = torch.empty_like(stack)  # Create an empty tensor with the same shape and device as stack
    new_stack[0, :, :] = new_frame       # Insert the new frame at the beginning of the stack
    new_stack[1:, :, :] = stack[:-1, :, :]  # Shift the rest of the frames

    return new_stack

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.data = deque()
        self.buffer_size = buffer_size

    def insert_data(self, state, action, reward, next_state, terminated, truncated):
        self.data.append((state, action, reward, next_state, terminated, truncated))

    def sample_data(self, num_of_data):
        return random.sample(self.data, min(len(self.data), num_of_data))
    
    def update_replay_buffer(self):
        if len(self.data) > self.buffer_size:
            self.data.popleft()
    
    def get_length(self):
        return len(self.data)