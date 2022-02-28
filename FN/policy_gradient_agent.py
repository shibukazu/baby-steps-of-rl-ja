import os
import random
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.python import keras as K
import gym
from fn_framework import FNAgent, Trainer, Observer, Experience
tf.compat.v1.disable_eager_execution()


class PolicyGradientAgent(FNAgent):

    def __init__(self, actions):
        # 方策勾配では必ず方策に従うのでepsilonは0
        super().__init__(epsilon=0.0, actions=actions)
        self.estimate_probs = True
        self.scaler = StandardScaler()
        self._updater = None

    def save(self, model_path):
        super().save(model_path)
        joblib.dump(self.scaler, self.scaler_path(model_path))

    @classmethod
    def load(cls, env, model_path):
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        agent.initialized = True
        agent.scaler = joblib.load(agent.scaler_path(model_path))
        return agent

    def scaler_path(self, model_path):
        fname, _ = os.path.splitext(model_path)
        fname += "_scaler.pkl"
        return fname

    def initialize(self, experiences, optimizer):
        states = np.vstack([e.s for e in experiences])
        feature_size = states.shape[1]
        self.model = K.models.Sequential([
            K.layers.Dense(10, activation="relu", input_shape=(feature_size,)),
            K.layers.Dense(10, activation="relu"),
            K.layers.Dense(len(self.actions), activation="softmax")
        ])
        self.set_updater(optimizer)
        self.scaler.fit(states)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def set_updater(self, optimizer):
        # actions: 現時点の方策に従った行動履歴すべて
        # rewards: 各行動時点より先の割引報酬和R_t（精度最大の見積もり価値）
        actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
        rewards = tf.compat.v1.placeholder(shape=(None), dtype="float32")
        # self.actions : とりうる行動一覧（one-hotの次元になる）
        # one_hot_actions : actionsのone-hot化
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1)
        # 現時点の行動確率（方策）
        action_probs = self.model.output
        # selected_action_probs : 行動履歴における各行動が対応する行動確率 \pi_\theta(a|s)
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs,
                                              axis=1)
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        # - log \pi_\theta(a|s) * Q^{\pi_\theta}(s, a) を計算
        loss = - tf.math.log(clipped) * rewards
        # E[- log \pi_\theta(a|s) * Q^{\pi_\theta}(s, a)] を計算
        # これが誤差関数になる
        loss = tf.reduce_mean(loss)
        # 上記誤差関数を利用して勾配逆伝搬法を行う
        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)
        self._updater = K.backend.function(
            inputs=[self.model.input,
                    actions, rewards],
            outputs=[loss],
            updates=updates)

    def estimate(self, s):
        normalized = self.scaler.transform(s)
        action_probs = self.model.predict(normalized)[0]
        return action_probs

    def update(self, states, actions, rewards):
        normalizeds = self.scaler.transform(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        self._updater([normalizeds, actions, rewards])


class CartPoleObserver(Observer):

    def transform(self, state):
        return np.array(state).reshape((1, -1))


class PolicyGradientTrainer(Trainer):

    def __init__(self, buffer_size=256, batch_size=32, gamma=0.9,
                 report_interval=10, log_dir=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)

    def train(self, env, episode_count=220, initial_count=-1, render=False):
        actions = list(range(env.action_space.n))
        agent = PolicyGradientAgent(actions)
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def episode_begin(self, episode, agent):
        if agent.initialized:
            self.experiences = []

    def make_batch(self, policy_experiences):
        length = min(self.batch_size, len(policy_experiences))
        batch = random.sample(policy_experiences, length)
        states = np.vstack([e.s for e in batch])
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        scaler = StandardScaler()
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = scaler.fit_transform(rewards).flatten()
        return states, actions, rewards

    def episode_end(self, episode, step_count, agent):
        # エピソード終了後の学習前処理（割引報酬和の計算など）
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        if not agent.initialized:
            if len(self.experiences) == self.buffer_size:
                optimizer = K.optimizers.Adam(lr=0.01)
                agent.initialize(self.experiences, optimizer)
                self.training = True
        else:
            policy_experiences = []
            for t, e in enumerate(self.experiences):
                s, a, r, n_s, d = e
                d_r = [_r * (self.gamma ** i) for i, _r in
                       enumerate(rewards[t:])]
                # 時刻tより先の割引報酬和R_tの計算
                d_r = sum(d_r)
                d_e = Experience(s, a, d_r, n_s, d)
                policy_experiences.append(d_e)
            # 学習を行う
            agent.update(*self.make_batch(policy_experiences))

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play):
    env = CartPoleObserver(gym.make("CartPole-v0"))
    trainer = PolicyGradientTrainer()
    path = trainer.logger.path_of("policy_gradient_agent.h5")

    if play:
        agent = PolicyGradientAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
