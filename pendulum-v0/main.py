import gym
import numpy as np
from playsound import playsound

from agent import DQNAgent
from stopwatch import Stopwatch

EPISODES = 1000
GOAL_SCORE = -500
GOAL_EPISODES = 10
GAME_LENGTH = 200
STATE_SHAPE = [1, 3]

# We can map from an action's index to the the torque value by going: index * 0.1 - 2.0
def action_index_to_torque(action_index):
    return [action_index * 0.1 - 2.0]

def play_a_game(env, agent, is_learning = True, is_rendering = False):
    state = env.reset()
    state = np.reshape(state, STATE_SHAPE)

    score = 0.0

    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time_t in range(GAME_LENGTH):
        # turn this on if you want to render
        if is_rendering:
            env.render()

        # Decide action
        action_index = agent.act(state)
        torque = action_index_to_torque(action_index)

        # Advance the game to the next frame based on the action.
        next_state, reward, done, _ = env.step(torque)
        next_state = np.reshape(next_state, STATE_SHAPE)

        # Update the score of the game based on the reward
        score += reward

        # Remember the previous state, action, reward, and done
        agent.remember(state, action_index, reward, next_state, done)

        # make next_state the new current state for the next frame.
        state = next_state

        if is_learning:
            agent.replay(32)

        if done:
            break

    return score

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('Pendulum-v0')
    state_size = env.observation_space.shape[0]    
    # The action size is going to be 41 which will correspond to: -2.0, -1.9, -1.8, ... 0.0, 0.1, 0.2, ... 2.0
    
    action_size = 41
    agent = DQNAgent(state_size, action_size)
    stopwatch = Stopwatch()

    achieved_goal_score_count = 0

    stopwatch.start()

    # Iterate the game
    for e in range(EPISODES):
        episode_number = e + 1

        # Play a game
        score = play_a_game(env, agent)

        # print the score
        print("episode: {}/{}, score: {}"
            .format(episode_number, EPISODES, score))

        # Determine if we've achieved our goal score
        if score >= GOAL_SCORE:
            achieved_goal_score_count += 1
        else:
            achieved_goal_score_count = 0
                
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

        score = play_a_game(env, agent, is_learning = False, is_rendering = True)

        print("episode: {}/{}, score: {}"
            .format(episode_number, 10, score))
            

