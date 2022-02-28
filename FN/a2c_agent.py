import argparse
from collections import deque
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python import keras as K
from PIL import Image
import gym
import gym_ple
from fn_framework import FNAgent, Trainer, Observer
tf.compat.v1.disable_eager_execution()


class ActorCriticAgent(FNAgent):

    def __init__(self, actions):
        # ActorCriticAgent uses self policy (doesn't use epsilon).
        super().__init__(epsilon=0.0, actions=actions)
        self._updater = None

    @classmethod
    def load(cls, env, model_path):
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path, custom_objects={
            "SampleLayer": SampleLayer})
        agent.initialized = True
        return agent

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.set_updater(optimizer)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=feature_shape,
            kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(256, kernel_initializer=normal,
                                 activation="relu"))
        # ここまで共有ネットワーク
        actor_layer = K.layers.Dense(len(self.actions),
                                     kernel_initializer=normal)
        # Q値を出力
        action_evals = actor_layer(model.output)
        # Q値に基づいて行動をサンプリング
        actions = SampleLayer()(action_evals)
        # V値を出力
        critic_layer = K.layers.Dense(1, kernel_initializer=normal)
        values = critic_layer(model.output)

        self.model = K.Model(inputs=model.input,
                             outputs=[actions, action_evals, values])

    def set_updater(self, optimizer,
                    value_loss_weight=1.0, entropy_weight=0.1):
        # ある方策にしたがった経験のバッチをつくり、学習を行う
        # actions: 方策に従い選択された行動
        # values: その行動の行動価値関数
        actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
        values = tf.compat.v1.placeholder(shape=(None), dtype="float32")
        # NNの出力(現在の状態に対する行動価値と状態価値)を取得
        _, action_evals, estimateds = self.model.output
        # softmax関数 + cross-entropy
        # softmax関数によってQ値から行動確率を計算
        # 実際にとった行動actionsをlabelとしてクロスエントロピーを求めていることから、
        # cross-entropyによって、実際にとった行動 a の -log(\pi_\theta(a|s))を求めている
        neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=action_evals, labels=actions)
        # advantageを計算する
        # tf.stop_gradient:
        #   advantageはactor側のlossの計算に使われており、そのままではactorの誤差がcriticにも伝搬してしまう
        #   この２つは目的関数が異なるからcritic側には逆伝搬させないようにする
        advantages = values - tf.stop_gradient(estimateds)
        # actor側の誤差を求める
        # E[-log(\pi_\theta(a|s) * A)]を計算
        policy_loss = tf.reduce_mean(neg_logs * advantages)
        # critic側の誤差を求める
        # TD誤差学習
        value_loss = tf.keras.losses.MeanSquaredError()(values, estimateds)

        action_entropy = tf.reduce_mean(self.categorical_entropy(action_evals))
        # 同時に学習するために２つのlossを合計したlossを全体のlossとする(critic側lossの寄与率をvalue_loss_weightで調整)
        loss = policy_loss + value_loss_weight * value_loss
        # 行動確率に基づくエントロピーを引くことで過学習を防ぐ
        # 行動確率にばらつきのある、エントロピーが大きい状態のほうが望ましいのでエントロピーが大きければ大きいほどlossが減少するようにする（正則化項）
        loss -= entropy_weight * action_entropy
        # 全体のlossに基づいて学習する
        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)

        self._updater = K.backend.function(
            inputs=[self.model.input,
                    actions, values],
            outputs=[loss,
                     policy_loss,
                     value_loss,
                     tf.reduce_mean(neg_logs),
                     tf.reduce_mean(advantages),
                     action_entropy],
            updates=updates)

    def categorical_entropy(self, logits):
        """
        From OpenAI baseline implementation.
        https://github.com/openai/baselines/blob/master/baselines/common/distributions.py#L192
        """
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def policy(self, s):
        if not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            action, action_evals, values = self.model.predict(np.array([s]))
            return action[0]

    def estimate(self, s):
        # 現在の状態をNNに入力したときに得られる状態価値のみを返す
        action, action_evals, values = self.model.predict(np.array([s]))
        return values[0][0]

    def update(self, states, actions, rewards):
        return self._updater([states, actions, rewards])


class SampleLayer(K.layers.Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1  # sample one action from evaluations
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)

    def call(self, x):
        noise = tf.random.uniform(tf.shape(x))
        return tf.argmax(x - tf.math.log(-tf.math.log(noise)), axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class ActorCriticAgentTest(ActorCriticAgent):

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Dense(10, input_shape=feature_shape,
                                 kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(10, kernel_initializer=normal,
                                 activation="relu"))

        actor_layer = K.layers.Dense(len(self.actions),
                                     kernel_initializer=normal)

        action_evals = actor_layer(model.output)
        actions = SampleLayer()(action_evals)

        critic_layer = K.layers.Dense(1, kernel_initializer=normal)
        values = critic_layer(model.output)

        self.model = K.Model(inputs=model.input,
                             outputs=[actions, action_evals, values])


class CatcherObserver(Observer):

    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # Convert the feature shape (f, w, h) => (h, w, f).
        feature = np.transpose(feature, (1, 2, 0))
        return feature


class ActorCriticTrainer(Trainer):

    def __init__(self, buffer_size=256, batch_size=32,
                 gamma=0.99, learning_rate=1e-3,
                 report_interval=10, log_dir="", file_name=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self.file_name = file_name if file_name else "a2c_agent.h5"
        self.learning_rate = learning_rate
        self.losses = {}
        self.rewards = []
        self._max_reward = -10

    def train(self, env, episode_count=900, initial_count=10,
              test_mode=False, render=False, observe_interval=100):
        actions = list(range(env.action_space.n))
        if not test_mode:
            agent = ActorCriticAgent(actions)
        else:
            agent = ActorCriticAgentTest(actions)
            observe_interval = 0
        self.training_episode = episode_count

        self.train_loop(env, agent, episode_count, initial_count, render,
                        observe_interval)
        return agent

    def episode_begin(self, episode, agent):
        self.rewards = []

    def step(self, episode, step_count, agent, experience):
        # 環境との相互作用1回ごとに呼ばれる
        self.rewards.append(experience.r)
        if not agent.initialized:
            if len(self.experiences) < self.buffer_size:
                # Store experience until buffer_size (enough to initialize).
                return False

            optimizer = K.optimizers.Adam(lr=self.learning_rate,
                                          clipnorm=5.0)
            agent.initialize(self.experiences, optimizer)
            self.logger.set_model(agent.model)
            self.training = True
            self.experiences.clear()
        else:
            if len(self.experiences) < self.batch_size:
                # Store experience until batch_size (enough to update).
                return False
            # 蓄えたexperience(同一の方策(Q関数)に従う)をもとにbatchを作成
            batch = self.make_batch(agent)
            # batchをもとに学習を行う
            loss, lp, lv, p_ng, p_ad, p_en = agent.update(*batch)
            # Record latest metrics.
            self.losses["loss/total"] = loss
            self.losses["loss/policy"] = lp
            self.losses["loss/value"] = lv
            self.losses["policy/neg_logs"] = p_ng
            self.losses["policy/advantage"] = p_ad
            self.losses["policy/entropy"] = p_en
            self.experiences.clear()

    def make_batch(self, agent):
        states = []
        actions = []
        values = []
        experiences = list(self.experiences)
        states = np.array([e.s for e in experiences])
        actions = np.array([e.a for e in experiences])

        # 各行動のQ値を計算する
        # Q(s,a) = r(s,a) + \gamma * V(s_n)
        last = experiences[-1]
        # future : V(s_n) ただし、エピソード終了の場合即時報酬を利用する
        future = last.r if last.d else agent.estimate(last.n_s)
        for e in reversed(experiences):
            value = e.r
            if not e.d:
                # r + \gamma * V(s_n)
                value += self.gamma * future
            values.append(value)
            # V(s_n) = Q(s_n, a_n) と近似している？？
            # 実際に撮った行動がaだから状態価値といえそう
            future = value
        values = np.array(list(reversed(values)))

        scaler = StandardScaler()
        values = scaler.fit_transform(values.reshape((-1, 1))).flatten()

        return states, actions, values

    def episode_end(self, episode, step_count, agent):
        reward = sum(self.rewards)
        self.reward_log.append(reward)

        if agent.initialized:
            self.logger.write(self.training_count, "reward", reward)
            self.logger.write(self.training_count, "reward_max",
                              max(self.rewards))

            for k in self.losses:
                self.logger.write(self.training_count, k, self.losses[k])

            if reward > self._max_reward:
                agent.save(self.logger.path_of(self.file_name))
                self._max_reward = reward

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play, is_test):
    file_name = "a2c_agent.h5" if not is_test else "a2c_agent_test.h5"
    trainer = ActorCriticTrainer(file_name=file_name)
    path = trainer.logger.path_of(trainer.file_name)
    agent_class = ActorCriticAgent

    if is_test:
        print("Train on test mode")
        obs = gym.make("CartPole-v0")
        agent_class = ActorCriticAgentTest
    else:
        env = gym.make("Catcher-v0")
        obs = CatcherObserver(env, 80, 80, 4)
        trainer.learning_rate = 7e-5

    if play:
        agent = agent_class.load(obs, path)
        agent.play(obs, episode_count=10, render=True)
    else:
        trainer.train(obs, test_mode=is_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2C Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")
    parser.add_argument("--test", action="store_true",
                        help="train by test mode")

    args = parser.parse_args()
    main(args.play, args.test)
