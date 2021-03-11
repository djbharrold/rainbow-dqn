import gym
import numpy as np
from agents.rainbow import RainbowDQN


def lunarlander_loop(agent, n_episodes):
    env = gym.make("LunarLander-v2")
    for i in range(n_episodes):
        done = False
        score = 0
        state = env.reset()
        while not done:
            state = np.expand_dims(state, axis=0)
            action = agent.choose_action()
            state_, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, state_, done)
            agent.learn()
            state = state_
            score += reward
        print("Episode: %03d" % (i + 1), "\t\tScore: %.2f" % score)


if __name__ == "__main__":
    N_INPUTS = 8
    N_ACTIONS = 4
    N_EPISODES = 250

    agent = RainbowDQN(n_inputs=N_INPUTS, n_actions=N_ACTIONS)
    lunarlander_loop(agent=agent, n_episodes=N_EPISODES)
