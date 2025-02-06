import gymnasium as gym
import ale_py
import torch
import gymnasium as gym
from Util import *
from Model import Model, Agent
import matplotlib.pyplot as plt

device = torch.device("cpu")
env = gym.make("Boxing-v4", max_episode_steps=225)
observation, info = env.reset(seed=42)
LEARNING_RATE = 0.0025
NUM_OF_STACKED_FRAMES = 4
TRAINING_THRESHOLD = 5000
EPSILON_DECAY = 0.01
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.025
DISCOUNT_FACTOR = 0.99
NUM_OF_STEPS_TO_UPDATE = 100

action_value_function = Model(num_actions=env.action_space.n).to(device)
optimizer = torch.optim.RMSprop(action_value_function.parameters(), lr=LEARNING_RATE, momentum=0.95)
deep_q_agent = Agent(EPSILON_DECAY, INITIAL_EPSILON, FINAL_EPSILON, DISCOUNT_FACTOR, action_value_function, torch.nn.MSELoss(), optimizer, device)
replay_buffer = ReplayBuffer(TRAINING_THRESHOLD)
number_of_episodes = 1000
step_counter = 0
episode_scores = []

for ep in range(1, number_of_episodes+1):
    state = env.reset()
    done, truncated = False, False
    episode_score = 0
    current_state = torch.zeros((NUM_OF_STACKED_FRAMES, 84, 84), dtype=torch.float32, device=device)

    while not done and not truncated:
        previous_state = current_state

        action = deep_q_agent.select_action(env.action_space.n, current_state.unsqueeze(0))

        observation, reward, done, truncated, info = env.step(action)
        episode_score += reward
        reward = max(-1, min(reward, 1))

        downsized_observation = downscale_image(observation)
        downsized_observation_tensor = torch.tensor(downsized_observation, dtype=torch.float32, device=device)

        current_state = update_frame_stack(current_state, downsized_observation_tensor)

        replay_buffer.insert_data(previous_state, action, reward, current_state, done, truncated)
        replay_buffer.update_replay_buffer()

        batch_data = replay_buffer.sample_data(32)

        if replay_buffer.get_length() >= TRAINING_THRESHOLD:
            deep_q_agent.training_step(batch_data)
            del batch_data

    deep_q_agent.decrement_epsilon()
    episode_scores.append(episode_score)
    print(f"Episode {ep} has ended. Agent has scored {episode_score} for this episode")
    print(f"Current Epsilon value is {deep_q_agent.get_epsilon()}")

    if ep % NUM_OF_STEPS_TO_UPDATE == 0:
        save_model = deep_q_agent.get_model()
        torch.save(save_model, 'agent.pth')

average_scores = []
chunk_size = 100

for i in range(0, len(episode_scores), chunk_size):
    chunk = episode_scores[i:i + chunk_size]
    average_score = sum(chunk) / len(chunk)
    average_scores.append(average_score)

# Plotting the average scores
plt.plot(average_scores)
plt.title("Average Scores per 100 Episodes")
plt.xlabel("Chunk of 100 Episodes")
plt.ylabel("Average Score")
plt.savefig("Reward.png")

env.close()