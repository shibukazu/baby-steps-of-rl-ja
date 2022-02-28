def V(s, gamma=0.99):
    # 報酬は現在の状態でのみ決まる
    # 報酬の減衰率を gamma とする
    # greedy方策とする
    V = R(s) + gamma * max_V_on_next_state(s)
    return V


def R(s):
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


def max_V_on_next_state(s):
    # 次の状態の価値関数に遷移確率をかけたもののうち最大のものを求める
    # If game end, expected value is 0.
    if s in ["happy_end", "bad_end"]:
        return 0

    actions = ["up", "down"]
    # 次の状態の価値観数に遷移確率をかけたもの
    values = []
    for a in actions:
        transition_probs = transit_func(s, a)
        v = 0
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            # 価値関数が計算済みである必要
            v += prob * V(next_state)
        values.append(v)
    return max(values)


def transit_func(s, a):
    """
    Make next state by adding action str to state.
    ex: (s = 'state', a = 'up') => 'state_up'
        (s = 'state_up', a = 'down') => 'state_up_down'
    """

    actions = s.split("_")[1:]
    LIMIT_GAME_COUNT = 5
    HAPPY_END_BORDER = 4
    MOVE_PROB = 0.9

    def next_state(state, action):
        return "_".join([state, action])

    if len(actions) == LIMIT_GAME_COUNT:
        up_count = sum([1 if a == "up" else 0 for a in actions])
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        prob = 1.0
        return {state: prob}
    else:
        # まだ終了していない場合
        # ハイパーパラメーターのMOVE_PROBに基づき、遷移確率を与える
        opposite = "up" if a == "down" else "down"
        return {
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }


if __name__ == "__main__":
    print(V("state"))
    print(V("state_up_up"))
    print(V("state_down_down"))
