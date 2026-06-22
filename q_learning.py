
import gymnasium as gym
import numpy as np
import random
from PIL import Image
import os
import matplotlib.pyplot as plt

gamma=0.99
epsilon=0.1
alpha=0.5
episodes=1000

Q=np.zeros((48,4))

def epsilon_greedy(state):
    if random.random()<epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

episode_rewards=[]
episode_lengths=[]

for episode in range(episodes):
    env=gym.make("CliffWalking-v1")
    done=False
    state,_=env.reset()
    episode_len=0
    tot_reward=0
    while not done:
        action=epsilon_greedy(state)
        next_state,reward,terminated,truncated,_=env.step(action)
        done=terminated or truncated
        Q[state,action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action])
        state=next_state
        episode_len+=1
        tot_reward+=reward
    episode_rewards.append(tot_reward)
    episode_lengths.append(episode_len)
    print(f"episode:{episode+1}/{episodes} total reward={tot_reward} and episode length={episode_len}")
    env.close()

window=100
moving_avg=np.convolve(episode_rewards,np.ones(window)/window,mode="valid")
os.makedirs("media",exist_ok=True)

plt.figure(figsize=(10,5))
plt.plot(episode_rewards,alpha=0.4,label="Episode Reward")
plt.plot(range(window-1,episodes),moving_avg,linewidth=2,label="Moving Average")
plt.title("Q-Learning on Cliff Walking")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.savefig("media/qlearning_rewards.png")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(episode_lengths)
plt.title("Episode Length")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)
plt.savefig("media/qlearning_steps.png")
plt.show()

env=gym.make("CliffWalking-v1",render_mode="rgb_array")
frames=[]
state,_=env.reset()
done=False
frames.append(env.render())
total_reward=0
steps=0
while not done:
    action=np.argmax(Q[state])
    state,reward,terminated,truncated,_=env.step(action)
    frames.append(env.render())
    done=terminated or truncated
    total_reward+=reward
    steps+=1
env.close()

print("\nFinal Evaluation")
print("----------------")
print("Reward :",total_reward)
print("Steps  :",steps)

images=[Image.fromarray(frame) for frame in frames]
images[0].save("media/qlearning.gif",save_all=True,append_images=images[1:],duration=350,loop=0)

print("\nGIF saved to media/qlearning.gif")
print("\nTraining Statistics")
print("----------------")
print("Average reward (last 100 episodes):",np.mean(episode_rewards[-100:]))
print("Average steps (last 100 episodes):",np.mean(episode_lengths[-100:]))
print("Best reward:",max(episode_rewards))
print("Worst reward:",min(episode_rewards))

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
