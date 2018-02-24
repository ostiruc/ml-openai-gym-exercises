import gym
import numpy as np
from playsound import playsound

from agent import DQNAgent
from stopwatch import Stopwatch

EPISODES = 1000
GOAL_SCORE = 200
GOAL_EPISODES = 10
STATE_SHAPE = [1, 3]

def action_index_to_torque(action_index):
    return [action_index * 0.2 - 2.0]

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('Pendulum-v0')
    state_size = env.observation_space.shape[0]    
    # The action size is going to be 21 which will correspond to: -2.0, -1.8, -1.6, ... 0.0, 0.2, 0.4, ... 2.0
    # We can map from an action's index to the the torque value by going: index * 0.2 - 2.0
    action_size = 21 
    agent = DQNAgent(state_size, action_size)
    stopwatch = Stopwatch()

    achieved_goal_score_count = 0

    stopwatch.start()
    # Iterate the game
    for e in range(EPISODES):
        episode_number = e + 1

        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, STATE_SHAPE)

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            # env.render()

            # Decide action
            action_index = agent.act(state)
            torque = action_index_to_torque(action_index)

            # Advance the game to the next frame based on the action.
            # Reward is TODO???
            next_state, reward, done, _ = env.step(torque)
            next_state = np.reshape(next_state, STATE_SHAPE)

            # Remember the previous state, action, reward, and done
            agent.remember(state, action_index, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            agent.replay(32)

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # TODO: Get the score working properly
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

    stopwatch.stop()

    playsound('./assets/work-complete.wav')
    print('Training completed in {} seconds.').format(stopwatch.total_run_time)
    raw_input('Press Enter to render some games...')

    # Play 10 more episodes and render them to show how awesome our agent is
    for e in range(10):
        episode_number = e + 1

        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, STATE_SHAPE)

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            env.render()

            # Decide action
            action_index = agent.act(state, force_exploitation=True)
            torque = action_index_to_torque(action_index)

            # Advance the game to the next frame based on the action.
            # Reward is TODO???
            next_state, reward, done, _ = env.step(torque)
            next_state = np.reshape(next_state, STATE_SHAPE)

            # No need to remember here as we are just showing off our agent.

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # TODO: Figure out the score for this problem
                score = time_t + 1

                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(episode_number, 10, score))
                break

