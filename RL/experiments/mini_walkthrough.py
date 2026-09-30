"""RL 三个十行以内的数值步骤：回报、Bellman 比较、一次 TD 更新。"""

import argparse


def return_lab() -> None:
    gamma = 0.9
    quit_now = 1.0
    invest_then_finish = -0.2 + gamma * 2.0
    print(f"回报：立即退出={quit_now:.1f}；先投入再完成={invest_then_finish:.1f}")
    assert invest_then_finish > quit_now


def bellman_lab() -> None:
    gamma = 0.9
    value_work = 2.0  # 在 work 直接完成得 2 分
    q_finish = 2.0
    q_wait = -0.1 + gamma * value_work
    q_invest = -0.2 + gamma * value_work
    print(f"Bellman：work 中完成={q_finish:.1f}，等待={q_wait:.1f}；start 中投入={q_invest:.1f}")
    assert (q_finish, q_wait, q_invest) == (2.0, 1.7, 1.6)


def update_lab() -> None:
    old_q = 0.0
    alpha = 0.5
    target = -0.2 + 0.9 * 2.0
    new_q = old_q + alpha * (target - old_q)
    print(f"一次 TD 更新：旧估计={old_q:.1f}，目标={target:.1f}，新估计={new_q:.1f}")
    assert new_q == 0.8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", choices=("return", "bellman", "update", "all"), nargs="?", default="all")
    chosen = parser.parse_args().lab
    for name, fn in (("return", return_lab), ("bellman", bellman_lab), ("update", update_lab)):
        if chosen in (name, "all"):
            fn()
