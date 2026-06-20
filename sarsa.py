import gymnasium as gym
import numpy as np
import random
from PIL import Image
import os

#create environment

env = gym.make("CliffWalking-v1")

type(env)

print(env.observation_space.n,env.action_space.n)
starting_state,_ = env.reset()

## implement SARSA

gamma = .99
alpha = 0.5
epsilon = 0.1
episodes = 500

# Q-TABLE -> STORE Q-values

Q = np.zeros((48,4))

#Policy - epsilon-greedy: state -> action
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample() #random action => EXPLORE
    else:
        return np.argmax(Q[state])  # exploit


for episode in range(episodes):


    env = gym.make("CliffWalking-v1")

    done = False # if reached End or got Truncate(cutshot -->exceeds limit)
    state,_ = env.reset()
    action = epsilon_greedy(state)

    total_reward = 0
    episode_length = 0

    while not done:
        next_state, reward, terminated, truncated, _= env.step(action)
        done = terminated or truncated
        next_action = epsilon_greedy(next_state)

        #SARSA UPDATE
        Q[state,action] += alpha * (reward + gamma*(Q[next_state,next_action]) - Q[state,action])

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1

    print(f"episode={episode}/500: tot reward={total_reward} and episode length ={episode_length}")
    env.close()


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
#
#     total_reward += reward
#     episode_len += 1
#
#
# print(f"total reward {total_reward} and episode length ={episode_len}")
# env.close()
