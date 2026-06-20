# Reinforcement Learning: Cliff Walking (Gymnasium)

Implementation of the classic **Cliff Walking** environment using two Temporal Difference (TD) reinforcement learning algorithms:

- **Q-Learning (Off-Policy)**
- **SARSA (On-Policy)**

The objective is to compare how both algorithms learn to navigate the environment while maximizing cumulative reward. Although both eventually reach the goal, they learn noticeably different policies because of their update mechanisms.

---

## Environment

The project uses Gymnasium's **CliffWalking-v1** environment.

The agent starts at the bottom-left corner and must reach the goal at the bottom-right corner.

- Stepping into the cliff gives a large negative reward and resets the agent.
- Every normal move receives a small negative reward.
- The objective is to maximize cumulative reward while reaching the goal efficiently.

---

# Algorithms

## Q-Learning

Q-Learning is an **off-policy** reinforcement learning algorithm.

It updates its Q-values using the maximum estimated value of the next state, allowing it to learn the optimal policy regardless of the action actually taken.

### Demonstration

<p align="center">
<img src="media/qlearning.gif" width="750">
</p>

---

## SARSA

SARSA is an **on-policy** reinforcement learning algorithm.

Instead of using the best possible next action, it updates the Q-value using the action that the agent actually follows according to its policy. This often produces safer behavior in risky environments such as Cliff Walking.

### Demonstration

<p align="center">
<img src="media/sarsa.gif" width="750">
</p>

---

# Hyperparameters

| Parameter | Value |
|-----------|------:|
| Episodes | 500 |
| Learning Rate (α) | 0.5 |
| Discount Factor (γ) | 0.99 |
| Exploration Rate (ε) | 0.1 |

---

# Q-Learning vs SARSA

| Feature | Q-Learning | SARSA |
|----------|------------|--------|
| Learning Type | Off-Policy | On-Policy |
| Update Rule | Uses max Q-value of next state | Uses next action actually taken |
| Exploration | More aggressive | More conservative |
| Learned Policy | Shortest path near the cliff | Safer path away from the cliff |
| Risk | Higher | Lower |

---

# Project Structure

```

Reinforcement-Learning-CliffWalking/
│
├── q_learning.py
├── sarsa.py
├── requirements.txt
├── LICENSE
├── README.md
└── media/
├── qlearning.gif
└── sarsa.gif

```

---

# Installation

Clone the repository

```bash
git clone https://github.com/ConquiMomo/Reinforcement-Learning-CliffWalking.git
```

Move into the project directory

```bash
cd Reinforcement-Learning-CliffWalking
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

Run the Q-Learning implementation

```bash
python q_learning.py
```

Run the SARSA implementation

```bash
python sarsa.py
```

---

# Requirements

- Python 3.10+
- Gymnasium
- NumPy
- Matplotlib

Install using

```bash
pip install -r requirements.txt
```

---

# Learning Outcomes

This project demonstrates:

- Reinforcement Learning fundamentals
- Temporal Difference Learning
- Q-Learning
- SARSA
- Exploration vs Exploitation
- Policy Learning
- Gymnasium Environment Interaction

---

# Future Improvements

- Plot cumulative reward over training episodes
- Compare convergence of both algorithms
- Tune hyperparameters
- Implement Expected SARSA
- Extend to larger GridWorld environments

---
