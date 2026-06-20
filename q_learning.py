import gymnasium as gym
import numpy as np
import random
from PIL import Image
import os


gamma = 0.99
epsilon = 0.1
alpha = 0.5
episodes =500
#%%
# Q-TABLE -> STORE Q-values

Q = np.zeros((48,4))
#%%
#epsilon-policy
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample() # explore
    else:
        return np.argmax(Q[state])
#%%
#Q-learning

for episode in range(episodes):

    env = gym.make("CliffWalking-v1")

    done = False
    state, _ =env.reset()

    episode_len = 0
    tot_reward =0

    while not done:
        action = epsilon_greedy(state)

        next_state, reward, terminated, truncated,_ = env.step(action)

        done = terminated or truncated

        #Q- LEARNING UPDATE
        Q[state,action]+=alpha*(reward + gamma*np.max(Q[next_state]) - Q[state,action])

        state = next_state
        episode_len +=1
        tot_reward +=reward

    print(f"episode:{episode}/500 total reward ={tot_reward} and episode length={episode_len}")
    env.close()


#creating gif for maximum reward

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

#saved gif

os.makedirs("media", exist_ok=True)

images = [Image.fromarray(frame) for frame in frames]

images[0].save(
    "media/qlearning.gif",
    save_all=True,
    append_images=images[1:],
    duration=350,
    loop=0,
)

print("\nGIF saved to media/qlearning.gif")

# #what did out agent learn
# env = gym.make("CliffWalking-v1",render_mode="human")
# state,_=env.reset()
# total_reward = 0
# episode_len = 0
# done = False
#
# while not done:
#     action = np.argmax(Q[state])
#     state,reward,terminated,truncated,_ = env.step(action)
#     done = terminated or truncated
#
#     episode_len += 1
#     total_reward+=reward
# print(f"total_reward ={total_reward}, total length = {episode_len}")
#
# env.close
