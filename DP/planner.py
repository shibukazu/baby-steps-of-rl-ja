class Planner():

    def __init__(self, env):
        self.env = env
        self.log = []

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        # ある状態・行動における全ての遷移確率・報酬の組を返す
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid


class ValueIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        # 可能な行動一覧
        actions = self.env.actions

        V = {}
        # 全状態の状態価値関数を適当な値で初期化
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        while True:
            delta = 0
            self.log.append(self.dict_to_grid(V))
            # 全状態に対して状態価値関数を更新する
            for s in V:
                # これ以上行動できなければスキップ
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                for a in actions:
                    r = 0
                    # 状態 s で行動 a をとった時の期待価値 r を求める（遷移先は確率的に定まるため、期待値を考える）
                    # greedy では全ての行動の中で最大の r を与えるものが実際の行動となり、そのときの r が状態価値になる
                    # このとき、今現在得られている状態価値関数に基づき計算する！
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                # 最大のものを今の状態の状態価値とする
                max_reward = max(expected_rewards)
                # 各状態に対する（現在の状態価値関数と更新後の状態価値関数の差の絶対値)をとり、そのうち最大のものを現在のステップの更新量とする
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward
            # 現在のステップの最大更新量が threshold を下回った場合、価値関数が十分な精度で求まったものとして、更新を終了する
            if delta < threshold:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid


class PolicyIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                # Initialize policy.
                # At first, each action is taken uniformly.
                self.policy[s][a] = 1 / len(actions)

    def estimate_by_policy(self, gamma, threshold):
        V = {}
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        # 現時点の方策に基づいて価値関数を反復的に更新し、精度を高め、求める
        # ここで計算される価値関数は現時点の方策の評価に利用される
        while True:
            delta = 0
            for s in V:
                # 状態sにおける各行動ごとの期待報酬をまとめたもの（これの和が）
                expected_rewards = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += action_prob * prob * \
                            (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            if delta < threshold:
                break

        return V

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions
        # 現時点の方策において最も選ばれやすいものを取り出す関数

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        while True:
            update_stable = True
            # 現時点の方策に基づき、状態価値関数の見積もりを更新する
            V = self.estimate_by_policy(gamma, threshold)
            self.log.append(self.dict_to_grid(V))

            for s in states:
                # 現在の方策に基づく、行動を取得
                policy_action = take_max_action(self.policy[s])

                # 新しい状態価値関数に基づいて、行動価値関数を計算する
                action_rewards = {}
                for a in actions:
                    r = 0
                    # 各行動ごとにすべての遷移先を考える
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    # 行動価値関数を保存する
                    action_rewards[a] = r
                # 行動価値関数に基づいた最適な行動の選択
                best_action = take_max_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                # 選ばれた最適な行動をするように方策を更新
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            if update_stable:
                # If policy isn't updated, stop iteration
                break

        # Turn dictionary to grid
        V_grid = self.dict_to_grid(V)
        return V_grid
