import random
from environment import Environment


class Agent():

    def __init__(self, env):
        # Environmentオブジェクトから可能な行動一覧を取得
        self.actions = env.actions

    def policy(self, state):
        # 方策はランダム方策
        return random.choice(self.actions)


def main():
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # Try 10 game.
    for i in range(10):
        # Initialize position of agent.
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 方策関数から行動を決定（実際に行動通りの方向へ移動できるかはわからない）
            action = agent.policy(state)
            # 遷移関数に基づき、次の遷移先を決定し、即時報酬を得る
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print("Episode {}: Agent gets {} reward.".format(i, total_reward))


if __name__ == "__main__":
    main()
