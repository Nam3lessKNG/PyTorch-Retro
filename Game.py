import gymnasium as gym
import pygame
import csv
from pettingzoo.butterfly import knights_archers_zombies_v10
import torch
from Util import *
from Model import Model, Agent
import os

device = torch.device("cpu")
env = knights_archers_zombies_v10.env(num_archers=4, num_knights=0, spawn_rate=10)
env.reset()

LEARNING_RATE = 0.001
MAX_STEPS = 1000
NUM_OF_STACKED_FRAMES = 4
TRAINING_THRESHOLD = 5000
EPSILON_DECAY = 0.001
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.01
DISCOUNT_FACTOR = 0.99
NUM_OF_STEPS_TO_UPDATE = 100
BATCH_SIZE = 256
NUM_OF_EPISODES = 100000

agents = env.possible_agents
agent_models = {}
agent_replay_buffers = {}

# Initialize models, optimizers, and replay buffers for each agent
for agent_id in agents:
    action_value_function = Model(num_actions=env.action_space(agent_id).n).to(device)
    optimizer = torch.optim.RMSprop(action_value_function.parameters(), lr=LEARNING_RATE, momentum=0.95)
    agent_instance = Agent(EPSILON_DECAY, INITIAL_EPSILON, FINAL_EPSILON, DISCOUNT_FACTOR, action_value_function, torch.nn.MSELoss(), optimizer, device)
    replay_buffer = ReplayBuffer(TRAINING_THRESHOLD)

    if os.path.exists(f"saves/{agent_id}.pth"):
        agent_instance.action_value_function.load_state_dict(torch.load(f"saves/{agent_id}.pth"))

    agent_models[agent_id] = agent_instance
    agent_replay_buffers[agent_id] = replay_buffer

episode_scores = []

for ep in range(1, NUM_OF_EPISODES + 1):
    env.reset()
    step_count = 0
    done = {agent_id: False for agent_id in agents}
    truncated = {agent_id: False for agent_id in agents}
    episode_score = {agent_id: 0 for agent_id in agents}
    
    current_states = {
        agent_id: torch.zeros((NUM_OF_STACKED_FRAMES, 84, 84), dtype=torch.float32, device=device)
        for agent_id in agents
    }

    while not all(done[agent] or truncated[agent] for agent in agents):
        step_count += 1
        if step_count > MAX_STEPS:
            print(f"Force exiting loop at step {step_count}. Done: {done}, Truncated: {truncated}")
            break

        for agent_id in env.agent_iter():
            
            # Get the latest observation and check if the agent is done
            observation, reward, done[agent_id], truncated[agent_id], _ = env.last()

            # If agent is dead, pass the turn
            if done[agent_id] or truncated[agent_id]:  
                env.step(None)
                continue  

            previous_state = current_states[agent_id]
            action = agent_models[agent_id].select_action(env.action_space(agent_id).n, previous_state.unsqueeze(0))
            env.step(action)
            episode_score[agent_id] += reward

            downsized_observation = downscale_image(observation)
            downsized_observation_tensor = torch.tensor(downsized_observation, dtype=torch.float32, device=device)

            current_states[agent_id] = update_frame_stack(current_states[agent_id], downsized_observation_tensor)

            agent_replay_buffers[agent_id].insert_data(previous_state, action, reward, current_states[agent_id], done[agent_id], truncated[agent_id])

            if agent_replay_buffers[agent_id].get_length() >= TRAINING_THRESHOLD:
                batch_data = agent_replay_buffers[agent_id].sample_data(BATCH_SIZE)
                agent_models[agent_id].training_step(batch_data)
                agent_replay_buffers[agent_id].prune()
                del batch_data

    for agent_id in agents:
        agent_models[agent_id].decrement_epsilon()

    episode_scores.append(sum(episode_score.values()))
    print(f"Episode {ep} ended. Total score: {sum(episode_score.values())}")

    if ep % NUM_OF_STEPS_TO_UPDATE == 0:
        for agent_id in agents:
            torch.save(agent_models[agent_id].action_value_function.state_dict(), f"saves/{agent_id}.pth")

csv_filename = "episode_rewards.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Total Reward"])
    for episode, score in enumerate(episode_scores, start=1):
        writer.writerow([episode, score])

env.close()