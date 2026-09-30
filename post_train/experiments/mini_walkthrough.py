"""后训练小算例：答案损失、偏好比较、DPO、组内优势。"""

import argparse
import math


def sft_lab() -> None:
    answer_prob = 0.6  # 给定题目后，正确答案 token 的模型概率
    loss = -math.log(answer_prob)
    print(f"SFT：正确答案概率={answer_prob:.1f}，负对数损失={loss:.6f}")
    assert 0.5 < loss < 0.6


def preference_lab() -> None:
    good_reward, bad_reward = 1.0, 0.0
    probability = 1.0 / (1.0 + math.exp(-(good_reward - bad_reward)))
    print(f"偏好模型：好答案胜出概率={probability:.6f}")
    assert 0.73 < probability < 0.74


def dpo_lab() -> None:
    chosen_new, rejected_new = -0.2, -1.2
    chosen_ref, rejected_ref = -0.5, -1.0
    beta = 0.2
    margin = beta * ((chosen_new - rejected_new) - (chosen_ref - rejected_ref))
    loss = math.log1p(math.exp(-margin))
    print(f"DPO：新旧相对概率差={margin:.3f}，损失={loss:.6f}")
    assert abs(margin - 0.1) < 1e-12


def group_lab() -> None:
    rewards = (1.0, 0.0, 0.0, 1.0)
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    advantage = tuple((r - mean) / std for r in rewards)
    print("GRPO：同题四份答案的相对优势=", advantage)
    assert advantage == (1.0, -1.0, -1.0, 1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", choices=("sft", "preference", "dpo", "group", "all"), nargs="?", default="all")
    chosen = parser.parse_args().lab
    for name, fn in (("sft", sft_lab), ("preference", preference_lab), ("dpo", dpo_lab), ("group", group_lab)):
        if chosen in (name, "all"):
            fn()
