import math
from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        # Q関数
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))
        # episode_count エピソードの間強化学習を行う（各エピソードごとに状態はリセットされる（環境自体は同じ？））
        for e in range(episode_count):
            # 初期状態を取得する
            s = env.reset()
            done = False
            # Play until the end of episode.
            experience = []
            while not done:
                if render:
                    env.render()
                # Q関数に対するepsilon-greedy で行動を決定
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
            else:
                self.log(reward)

            # 各経験における状態と価値のQ関数を、それを経験した時点からエピソード終了時点までの残りの経験を利用して更新する
            for i, x in enumerate(experience):
                # 更新対象の状態と行動のセットを取得する
                # Q(s,a)が更新される
                s, a = x["state"], x["action"]

                # 経験時より後の残りの経験を利用してQ関数を更新する
                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1
                # alphaの計算の仕方の理由はよくわからん
                N[s][a] += 1  # count of s, a pair
                alpha = 1 / N[s][a]
                # G - Q(s,a) : TD誤差
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=500)
    show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
