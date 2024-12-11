# CartPole Q-Learning

This project implements a Q-learning algorithm to solve the CartPole environment from OpenAI's Gym. The agent learns to balance the cart by interacting with the environment and adjusting its policy over multiple episodes. The Q-table is used to store the expected future rewards for state-action pairs, and the epsilon-greedy policy is applied to balance exploration and exploitation.

[![Watch the video]([https://raw.githubusercontent.com/DDursun/Cart-Pole-Balancing/main/videos/thumbnail.jpg](https://github.com/DDursun/Cart-Pole-Balancing/blob/8213922c35da31d9e4bcc918e9370edf7e2f3714/videos/thumbnail.png))]([https://raw.githubusercontent.com/DDursun/Cart-Pole-Balancing/main/videos/rl-video-episode-1.mp4](https://github.com/DDursun/Cart-Pole-Balancing/raw/refs/heads/main/videos/rl-video-episode-1.mp4))


## Requirements

- Python 3.6+
- Gymnasium
- Numpy

You can install the necessary dependencies using:

```bash
pip install gymnasium numpy
