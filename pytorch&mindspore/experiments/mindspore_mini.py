"""与 torch_mini.py 对齐的 MindSpore 2.4 教学小实验；需自行安装适配的 MindSpore。"""

import argparse
import mindspore as ms
from mindspore import ops


def scalar(tensor) -> float:
    return float(tensor.asnumpy().item())


def shape_lab() -> None:
    prediction = ms.Tensor([[1.0], [2.0], [3.0]], ms.float32)
    target = ms.Tensor([1.0, 2.0, 3.0], ms.float32)
    wrong = prediction - target
    right = prediction[:, 0] - target
    print("形状实验：错误差值形状", wrong.shape, "MSE", scalar(ops.mean(wrong ** 2)))
    print("形状实验：正确差值形状", right.shape, "MSE", scalar(ops.mean(right ** 2)))
    assert wrong.shape == (3, 3) and right.shape == (3,)


def gradient_lab() -> None:
    def loss_fn(w, b):
        return (2.0 * w + b - 5.0) ** 2 / 2.0

    w = ms.Tensor(1.0, ms.float32)
    b = ms.Tensor(0.0, ms.float32)
    loss, (dw, db) = ms.value_and_grad(loss_fn, grad_position=(0, 1))(w, b)
    print("梯度实验：损失", scalar(loss), "dw", scalar(dw), "db", scalar(db))
    assert (scalar(dw), scalar(db)) == (-6.0, -3.0)
    w = w - 0.1 * dw
    b = b - 0.1 * db
    print("一步更新：w", scalar(w), "b", scalar(b), "新预测", scalar(2.0 * w + b))


def fit_lab() -> None:
    x = ms.Tensor([0.0, 1.0, 2.0], ms.float32)
    y = ms.Tensor([1.0, 3.0, 5.0], ms.float32)

    def loss_fn(w, b):
        return ops.mean((w * x + b - y) ** 2)

    loss_and_grad = ms.value_and_grad(loss_fn, grad_position=(0, 1))
    w = ms.Tensor(0.0, ms.float32)
    b = ms.Tensor(0.0, ms.float32)
    for step in range(201):
        loss, (dw, db) = loss_and_grad(w, b)
        if step in (0, 20, 200):
            print(f"训练第 {step:3d} 步：MSE={scalar(loss):.6f}，w={scalar(w):.4f}，b={scalar(b):.4f}")
        if step == 200:
            break
        w = w - 0.1 * dw
        b = b - 0.1 * db
    print("新输入 x=3：预测", scalar(3.0 * w + b), "目标 7")
    assert scalar(loss) < 1e-6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", choices=("shape", "gradient", "fit", "all"), nargs="?", default="all")
    lab = parser.parse_args().lab
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    for name, fn in (("shape", shape_lab), ("gradient", gradient_lab), ("fit", fit_lab)):
        if lab in (name, "all"):
            fn()
