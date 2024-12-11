import gymnasium as gym
import numpy as np

####### VVV Don't modify this part VVV #######

np.random.seed(1)

observation_grid_size = [30, 24, 30, 30]
observation_grid_middle = np.array([g/2 for g in observation_grid_size])
observation_discretization_width = np.array([0.16, 0.15, 0.014, 0.05])

def discretize_state(s):
	idxs = (s/observation_discretization_width + observation_grid_middle).astype(np.int32)
	for i in range(4):
		idxs[i] = max(min(idxs[i], observation_grid_size[i]-1), 0)
	return tuple(idxs)

def simulate_and_render(q_table, n_reps):
	env = gym.make('CartPole-v1', render_mode='rgb_array')
	env = gym.wrappers.RecordVideo(env, f'videos', episode_trigger=lambda x: True)
	for rep in range(n_reps):
		state = discretize_state(env.reset()[0])
		done = False
		rewards = 0.
		episode_step = 0
		while not done:
			episode_step += 1
			action = np.argmax(q_table[state])
			state, reward, done, _, _ = env.step(action)
			state = discretize_state(state)
			rewards += reward
			done = done or (episode_step >= 1000)
		print(f'Total reward for try #{rep}: {rewards}')
	env.close()

env = gym.make("CartPole-v1")
q_table = np.random.uniform(low=0, high=1, size=(observation_grid_size + [env.action_space.n]))

def main():

    ####### VVV Set hyperparameters VVV #######
    n_episodes = 30000
    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.01
    lr = 0.1
    ####### ^^^ Set hyperparameters ^^^ #######

    ####### Your code goes below #######
    for episode in range(n_episodes):
        state = discretize_state(env.reset()[0])
        done = False
        total_reward = 0

        while not done:
            # Exploration
            if np.random.random() < epsilon:
                action = env.action_space.sample() 
            
            # Exploitaton
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state)

            # Updating q values
            best_next_action = np.max(q_table[next_state])
            expected_return  = reward + gamma * best_next_action * (not done)
            td_error = expected_return  - q_table[state][action]
            q_table[state][action] += lr * td_error

            state = next_state
            total_reward += reward

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon = 1.0 - episode / (n_episodes*0.85)

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    # Close the environment and simulate
    env.close()
    simulate_and_render(q_table, n_reps=3)


    # env.reset() restarts the game environment
    # this needs to be called at the start of each episode
    # its first return value is the initial state (in continuous form)
    continuous_state,_ = env.reset()

    # The provided function discretize_state converts the continuous
    # state into the index of the corresponding discretized state
    discrete_state = discretize_state(continuous_state)
    # You can use the discretized state in order to access the q values
    # q_table[discretize_state] -> [Q(s,0), Q(s,1)]

    # Once you have selected an action (either 0 or 1), you can play that action
    # using env.step(), which returns the next continuous state, the reward 
    # earned by the agent, and ``terminated'' which is a boolean value indicating
    # whether the game has ended (which happens if the cart moves too far left
    # or right, or if the pole tips too far left or right). When terminated==True
    # the episode is over and you'll need to start over by calling env.reset()
    new_continuous_state, reward, terminated, _, _ = env.step(action)

    # at the very end, you can call env.close() to clean everything up
    env.close()

    # This function is provided to simulate your agent, as described by q_table,
    # n_reps times. It also renders a video of each of your agent's attempts.
    simulate_and_render(q_table, n_reps=3)



if __name__ == "__main__":
    main()




