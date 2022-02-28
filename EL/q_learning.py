from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class QLearningAgent(ELAgent):
    # TD学習の実装

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        # 各エピソードごとの学習
        for e in range(episode_count):
            s = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                # 実際の行動価値(Q学習では遷移先の状態の価値V(s)=max(Q(s,a))としている)
                gain = reward + gamma * max(self.Q[n_state])
                # 現時点の見積もり行動価値
                estimated = self.Q[s][a]
                # 1stepごとにTD誤差で学習する
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state

            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    agent = QLearningAgent()
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=500)
    show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
