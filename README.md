

This project implements a **Deep Q-Learning (DQN) agent** to solve the **CartPole-v1** environment from **OpenAI Gym** using **TensorFlow**. The goal is to train an agent that can balance a pole on a moving cart by learning from interactions with the environment.

---

### Features

* Deep Q-Network (DQN) with experience replay.
* Epsilon-greedy exploration strategy.
* Trains and evaluates on CartPole-v1.
* Modular codebase for easy extensions.

---

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/dqn_cartpole.git
   cd dqn_cartpole
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

**Train the agent:**

```bash
python train.py
```

This will start the training process and print the progress to the console.

**Evaluate the trained agent:**

```bash
python evaluate.py
```

This will run the agent in the CartPole environment to test its performance.

---

### Requirements

* TensorFlow
* Gym
* NumPy
* Matplotlib (for plotting rewards)

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

---

### Notes

* The agent saves the trained model after training.
* You can adjust hyperparameters (like learning rate, epsilon decay) in `agent.py`.
* The project uses `CartPole-v1`, which requires balancing for 500 steps to be considered solved.
