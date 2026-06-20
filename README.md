# Reinforcement Learning - Cliff Walking

Implementation of the classic **Cliff Walking** problem using two reinforcement learning algorithms:

- Q-Learning (Off-Policy)
- SARSA (On-Policy)

## Environment

The agent must move from the **Start** state to the **Goal** state while avoiding the cliff.

- Reward for reaching Goal: Positive
- Reward for falling into Cliff: Large Negative
- Objective: Learn the optimal policy.

---

## Algorithms Implemented

### 1. Q-Learning
- Off-policy TD Control
- Learns the optimal policy regardless of the behavior policy

<p align="center">
  <img src="media/qlearning.gif" width="700" alt="Q-Learning Cliff Walking">
</p>

---

### 2. SARSA
- On-policy TD Control
- Learns using the action actually taken

<p align="center">
  <img src="media/sarsa.gif" width="700" alt="SARSA Cliff Walking">
</p>

---

## Project Structure

```
Reinforcement-Learning-CliffWalking/
│
├── Cliff Walking problem using Q-learning.py
├── Cliff Walking problem using SARSA.py
├── README.md
└── media/
    ├── qlearning.gif
    └── sarsa.gif
```

---

## Requirements

```bash
pip install numpy matplotlib
```

---

## Run

```bash
python "Cliff Walking problem using Q-learning.py"
```

or

```bash
python "Cliff Walking problem using SARSA.py"
```

---

## Concepts Covered

- Reinforcement Learning
- Temporal Difference Learning
- Q-Learning
- SARSA
- Exploration vs Exploitation
- Cliff Walking Environment
