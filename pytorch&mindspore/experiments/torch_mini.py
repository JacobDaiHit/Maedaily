"""三个可以单独运行的 PyTorch 小实验：形状、梯度、训练。"""

import argparse
import torch


def shape_lab() -> None:
    prediction = torch.tensor([[1.0], [2.0], [3.0]])  # [3, 1]
    target = torch.tensor([1.0, 2.0, 3.0])          # [3]
    wrong = prediction - target                       # 广播成 [3, 3]
    right = prediction[:, 0] - target                 # 对齐成 [3]
    print("形状实验：错误差值形状", tuple(wrong.shape), "MSE", wrong.square().mean().item())
    print("形状实验：正确差值形状", tuple(right.shape), "MSE", right.square().mean().item())
    assert wrong.shape == (3, 3) and right.shape == (3,)


def gradient_lab() -> None:
    w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    prediction = 2.0 * w + b
    loss = 0.5 * (prediction - 5.0).square()
    loss.backward()
    print("梯度实验：预测", prediction.item(), "损失", loss.item())
    print("梯度实验：dw", w.grad.item(), "db", b.grad.item())
    assert (w.grad.item(), b.grad.item()) == (-6.0, -3.0)
    with torch.no_grad():
        w -= 0.1 * w.grad
        b -= 0.1 * b.grad
    print("一步更新：w", w.item(), "b", b.item(), "新预测", (2.0 * w + b).item())


def fit_lab() -> None:
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = 2.0 * x + 1.0
    w = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    for step in range(201):
        loss = ((w * x + b - y) ** 2).mean()
        if step in (0, 20, 200):
            print(f"训练第 {step:3d} 步：MSE={loss.item():.6f}，w={w.item():.4f}，b={b.item():.4f}")
        if step == 200:
            break
        loss.backward()
        with torch.no_grad():
            w -= 0.1 * w.grad
            b -= 0.1 * b.grad
            w.grad = None
            b.grad = None
    with torch.no_grad():
        print("新输入 x=3：预测", (3.0 * w + b).item(), "目标 7")
    assert loss.item() < 1e-8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", choices=("shape", "gradient", "fit", "all"), nargs="?", default="all")
    lab = parser.parse_args().lab
    torch.set_num_threads(1)
    for name, fn in (("shape", shape_lab), ("gradient", gradient_lab), ("fit", fit_lab)):
        if lab in (name, "all"):
            fn()
