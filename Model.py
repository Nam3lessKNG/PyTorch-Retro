import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

# The model is a DQN that uses PyTorch. The numbers in the conv2d is for the size of the game and environment
# The fc1 is the same thing with the (7 * 7 * 64) which is the size of the env AFTER we downscale the frame
# This was also the combinaiton of other models found online
class Model(nn.Module):
    def __init__(self, num_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) 
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# The agent class is supposed to hold all the hyper parameters, select the actions, and train the model
class Agent:
    def __init__(self, epsilon_decay, initial_epsilon, final_epsilon, discount_factor, action_value_function, loss_fn, optimizer, device):
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.action_value_function = action_value_function.to(device)
        self.target_action_value_function = copy.deepcopy(action_value_function).to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epsilon = initial_epsilon
        self.training_step_counter = 0
        self.action_counter = 0
        self.previous_action = None
        self.device = device

    # The action is greedy epsilon method where it will commit an action with a percentage chance of it being random
    # Doing this makes it so that it has some data to learn from the start and then progress further into the run 
    def select_action(self, action_space, current_state):
        possible_actions = list(range(action_space))

        if self.action_counter % 4 == 0 or self.previous_action is None:
            if random.random() < self.epsilon:
                chosen_action = random.choice(possible_actions)
            else:
                with torch.no_grad():
                    q_values = self.action_value_function(current_state)
                    chosen_action = torch.argmax(q_values).item()
            self.previous_action = chosen_action
            self.action_counter = 0
        else:
            chosen_action = self.previous_action

        self.action_counter += 1
        return chosen_action

    # Here is when we take the batch from the replay buffer and train off of PyTorch tensors.
    def training_step(self, minibatch):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminated, batch_truncated = zip(*minibatch)
        
        # We run the batch into the tensors and get back their weight
        batch_states = torch.stack(batch_states).to(self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
        batch_next_states = torch.stack(batch_next_states).to(self.device)
        batch_terminated = torch.tensor(batch_terminated, dtype=torch.bool, device=self.device)
        batch_truncated = torch.tensor(batch_truncated, dtype=torch.bool, device=self.device)

        q_values = self.action_value_function(batch_states)
        current_q = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

        # Handle done conditions
        not_done_mask = ~ (batch_terminated | batch_truncated)
        next_q_values = self.target_action_value_function(batch_next_states).detach()
        next_q_max = next_q_values.max(1)[0]
        batch_target = batch_rewards + self.discount_factor * next_q_max * not_done_mask

        # handle loss and gradient
        loss = self.loss_fn(current_q, batch_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update of target network 
        tau = 0.005 
        for target_param, param in zip(self.target_action_value_function.parameters(), self.action_value_function.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.training_step_counter += 1

    def decrement_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def get_epsilon(self):
        return self.epsilon
    
    def get_model(self):
        return self.action_value_function