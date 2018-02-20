import gym
import numpy as np
from playsound import playsound

from agent import DQNAgent

EPISODES = 1000
GOAL_SCORE = 200
GOAL_EPISODES = 5

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    achieved_goal_score_count = 0

    # Iterate the game
    for e in range(EPISODES):
        episode_number = e + 1

        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            # env.render()

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            agent.replay(32)

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                score = time_t + 1

                # print the score
                print("episode: {}/{}, score: {}"
                      .format(episode_number, EPISODES, score))

                # Determine if we've achieved our goal score
                if score >= GOAL_SCORE:
                    achieved_goal_score_count += 1
                else:
                    achieved_goal_score_count = 0

                # break out of the loop
                break

        # If we've met our goal score enough times in a row then end training
        if achieved_goal_score_count >= GOAL_EPISODES:
            print("Achieved Goal Episodes, Ending Training...")
            break

    playsound('./assets/work-complete.wav')
    raw_input('Training Complete, press Enter to render some games...')

    # Play 10 more episodes and render them to show how awesome our agent is
    for e in range(10):
        episode_number = e + 1

        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            env.render()

            # Decide action
            action = agent.act(state, force_exploitation=True)

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            # No need to remember here as we are just showing off our agent.

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                score = time_t + 1

                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(episode_number, 10, score))
                break

