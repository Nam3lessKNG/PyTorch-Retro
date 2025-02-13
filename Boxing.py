import gymnasium as gym
import ale_py
import torch
import gymnasium as gym
from Util import *
from Model import Model, Agent
import matplotlib.pyplot as plt
import os

device = torch.device("cpu")
env = gym.make("Boxing-v4", max_episode_steps=225)
observation, info = env.reset(seed=42)
LEARNING_RATE = 0.001
NUM_OF_STACKED_FRAMES = 4
TRAINING_THRESHOLD = 10000
EPSILON_DECAY = 0.001
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.01
DISCOUNT_FACTOR = 0.99
NUM_OF_STEPS_TO_UPDATE = 100

action_value_function = Model(num_actions=env.action_space.n).to(device)
optimizer = torch.optim.RMSprop(action_value_function.parameters(), lr=LEARNING_RATE, momentum=0.95)
agent = Agent(EPSILON_DECAY, INITIAL_EPSILON, FINAL_EPSILON, DISCOUNT_FACTOR, action_value_function, torch.nn.MSELoss(), optimizer, device)
replay_buffer = ReplayBuffer(TRAINING_THRESHOLD)
number_of_episodes = 5000
episode_scores = []

if(os.path.exists("agent.pth")):
    agent.action_value_function.load_state_dict(torch.load("agent.pth"))

for ep in range(1, number_of_episodes+1):
    state, info = env.reset()
    done, truncated = False, False
    episode_score = 0
    current_state = torch.zeros((NUM_OF_STACKED_FRAMES, 84, 84), dtype=torch.float32, device=device)

    while not done and not truncated:
        previous_state = current_state

        action = agent.select_action(env.action_space.n, current_state.unsqueeze(0))

        observation, reward, done, truncated, info = env.step(action)
        episode_score += reward

        downsized_observation = downscale_image(observation)
        downsized_observation_tensor = torch.tensor(downsized_observation, dtype=torch.float32, device=device)

        current_state = update_frame_stack(current_state, downsized_observation_tensor)

        replay_buffer.insert_data(previous_state, action, reward, current_state, done, truncated)

        if replay_buffer.get_length() >= TRAINING_THRESHOLD:
            batch_data = replay_buffer.sample_data(32)
            agent.training_step(batch_data)
            del batch_data

    agent.decrement_epsilon()
    episode_scores.append(episode_score)
    print(f"Episode {ep} has ended. Agent has scored {episode_score} for this episode")
    print(f"Current Epsilon value is {agent.get_epsilon()}")

    if ep % NUM_OF_STEPS_TO_UPDATE == 0:
        torch.save(agent.action_value_function.state_dict(), 'agent.pth')

average_scores = []
chunk_size = 250

for i in range(0, len(episode_scores), chunk_size):
    chunk = episode_scores[i:i + chunk_size]
    average_score = sum(chunk) / len(chunk)
    average_scores.append(average_score)

# Plotting the average scores
plt.plot(average_scores)
plt.title(f"Average Scores per {chunk_size} Episodes")
plt.xlabel(f"Chunk of {chunk_size} Episodes")
plt.ylabel("Average Score")
plt.savefig("Reward.png")

env.close()