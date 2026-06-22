import gymnasium as gym
import numpy as np
import random
from PIL import Image
import os
import matplotlib.pyplot as plt

#create environment

env = gym.make("CliffWalking-v1")

type(env)

print(env.observation_space.n, env.action_space.n)
starting_state, _ = env.reset()

## implement SARSA

gamma = .99
alpha = 0.5
epsilon = 0.1
episodes = 1000

# Q-TABLE -> STORE Q-values

Q = np.zeros((48,4))

#Policy - epsilon-greedy: state -> action
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()     # random action => EXPLORE
    else:
        return np.argmax(Q[state])           # exploit

episode_rewards = []
episode_lengths = []

for episode in range(episodes):

    env = gym.make("CliffWalking-v1")

    done = False
    state, _ = env.reset()
    action = epsilon_greedy(state)

    total_reward = 0
    episode_length = 0

    while not done:

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        next_action = epsilon_greedy(next_state)

        # SARSA UPDATE
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1

    episode_rewards.append(total_reward)
    episode_lengths.append(episode_length)

    print(f"episode={episode+1}/{episodes}: tot reward={total_reward} and episode length={episode_length}")

    env.close()

#Reward Graph

window = 100

moving_avg = np.convolve(
    episode_rewards,
    np.ones(window)/window,
    mode="valid"
)

os.makedirs("media", exist_ok=True)

plt.figure(figsize=(10,5))

plt.plot(
    episode_rewards,
    alpha=0.4,
    label="Episode Reward"
)

plt.plot(
    range(window-1, episodes),
    moving_avg,
    linewidth=2,
    label="Moving Average"
)

plt.title("SARSA on Cliff Walking")

plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.legend()
plt.grid(True)

plt.savefig("media/sarsa_rewards.png")

plt.show()

#Episode Length Graph

plt.figure(figsize=(10,5))

plt.plot(episode_lengths)

plt.title("Episode Length")

plt.xlabel("Episode")
plt.ylabel("Steps")

plt.grid(True)

plt.savefig("media/sarsa_steps.png")

plt.show()

#creating maximum reward as gif

env = gym.make("CliffWalking-v1", render_mode="rgb_array")

frames = []

state, _ = env.reset()

done = False

frames.append(env.render())

total_reward = 0
steps = 0

while not done:

    action = np.argmax(Q[state])

    state, reward, terminated, truncated, _ = env.step(action)

    frames.append(env.render())

    done = terminated or truncated

    total_reward += reward
    steps += 1

env.close()

print("\nFinal Evaluation")
print("----------------")
print("Reward :", total_reward)
print("Steps  :", steps)

#saving gif

os.makedirs("media", exist_ok=True)

images = [Image.fromarray(frame) for frame in frames]

images[0].save(
    "media/sarsa.gif",
    save_all=True,
    append_images=images[1:],
    duration=350,
    loop=0,
)

print("\nGIF saved to media/sarsa.gif")

#Training Statistics

print("\nTraining Statistics")
print("Average reward (last 100 episodes):",
      np.mean(episode_rewards[-100:]))

print("Average steps (last 100 episodes):",
      np.mean(episode_lengths[-100:]))

print("Best reward:", max(episode_rewards))
print("Worst reward:", min(episode_rewards))

# #WHAT DID OUR AGENT LEARN
#
# env = gym.make("CliffWalking-v1", render_mode="human")
# state,_= env.reset()
# done =False
# total_reward = 0
# episode_len = 0
#
# while not done:
#     action = np.argmax(Q[state])
#     state, reward, terminated,truncated,_ = env.step(action)
#     done = terminated or truncated
#
#     total_reward += reward
#     episode_len += 1
#
# print(f"total reward {total_reward} and episode length ={episode_len}")
# env.close()
