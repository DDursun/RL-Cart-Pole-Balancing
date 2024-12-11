# CartPole Q-Learning

This project implements a Q-learning algorithm to solve the CartPole environment from OpenAI's Gym. The agent learns to balance the cart by interacting with the environment and adjusting its policy over multiple episodes. The Q-table is used to store the expected future rewards for state-action pairs, and the epsilon-greedy policy is applied to balance exploration and exploitation.

With only 30k training iterations, the agent is able to balance the pole for 250+ timesteps:


https://github.com/user-attachments/assets/9ab1af2f-fe26-4a87-b9e9-4e6173fb3f95



## Requirements

- Python 3.6+
- Gymnasium
- Numpy

You can install the necessary dependencies using:

```bash
pip install gymnasium numpy
