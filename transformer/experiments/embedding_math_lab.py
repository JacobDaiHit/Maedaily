"""Embedding arithmetic: lookup, SGNS, pooling, contrastive loss and ranking.

Run from the repository root:
    python -X utf8 transformer/experiments/embedding_math_lab.py

Uses only the Python standard library. This checks calculations and performs
one explicit SGD step; it does not train or evaluate a text encoder.
"""

import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("点积的两个向量必须具有同一维度")
    return sum(x * y for x, y in zip(a, b))


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def sgns_loss(rows):
    center, positive, negative = rows
    return softplus(-dot(center, positive)) + softplus(dot(center, negative))


def sgns_gradients(rows):
    center, positive, negative = rows
    positive_factor = sigmoid(dot(center, positive)) - 1.0
    negative_factor = sigmoid(dot(center, negative))
    return [
        [positive_factor * p + negative_factor * n
         for p, n in zip(positive, negative)],
        [positive_factor * x for x in center],
        [negative_factor * x for x in center],
    ]


def central_gradients(rows, epsilon=1e-6):
    gradients = []
    for row_index, row in enumerate(rows):
        result = []
        for column_index in range(len(row)):
            plus = [list(values) for values in rows]
            minus = [list(values) for values in rows]
            plus[row_index][column_index] += epsilon
            minus[row_index][column_index] -= epsilon
            result.append((sgns_loss(plus) - sgns_loss(minus)) / (2 * epsilon))
        gradients.append(result)
    return gradients


def masked_mean(hidden, mask):
    if not hidden or len(hidden) != len(mask):
        raise ValueError("隐藏状态与 mask 必须具有相同的非零位置数")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("本实验 mask 只能包含 0 或 1")
    count = sum(mask)
    if count == 0:
        raise ValueError("pooling 至少需要一个有效位置")
    dimension = len(hidden[0])
    if any(len(row) != dimension for row in hidden):
        raise ValueError("隐藏状态的向量维度必须一致")
    return [sum(flag * row[j] for flag, row in zip(mask, hidden)) / count
            for j in range(dimension)]


def normalize(vector):
    norm = math.sqrt(dot(vector, vector))
    if norm == 0:
        raise ValueError("零向量没有余弦方向")
    return [value / norm for value in vector]


def contrastive_row(similarities, temperature):
    """One query, positive document at index 0, no batch mean factor."""
    if temperature <= 0:
        raise ValueError("温度必须大于零")
    logits = [value / temperature for value in similarities]
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    denominator = sum(weights)
    probabilities = [value / denominator for value in weights]
    loss = maximum + math.log(denominator) - logits[0]
    similarity_gradients = [(probability - (1 if j == 0 else 0)) / temperature
                            for j, probability in enumerate(probabilities)]
    return loss, probabilities, similarity_gradients


def close(actual, expected, tolerance=1e-9):
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(f"实际 {actual}，预期 {expected}，容差 {tolerance}")


def main():
    table = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    ids = [2, 1, 2]
    gradients = [[0.0, 0.0] for _ in table]
    for index in ids:
        for j, value in enumerate(table[index]):
            gradients[index][j] += value
    lookup_loss = 0.5 * sum(dot(table[index], table[index]) for index in ids)
    close(lookup_loss, 1.5)
    if gradients != [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]:
        raise AssertionError("重复编号的梯度没有正确累加")
    print(f"查表：损失={lookup_loss}，四行梯度={gradients}")

    rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    analytic = sgns_gradients(rows)
    numerical = central_gradients(rows)
    gradient_error = max(abs(a - n) for ar, nr in zip(analytic, numerical)
                         for a, n in zip(ar, nr))
    close(gradient_error, 0.0, tolerance=1e-8)
    initial_loss = sgns_loss(rows)
    close(initial_loss, 2.006408868078168)
    updated = [[value - 0.1 * gradient for value, gradient in zip(row, grad)]
               for row, grad in zip(rows, analytic)]
    final_loss = sgns_loss(updated)
    positive_score = dot(updated[0], updated[1])
    negative_score = dot(updated[0], updated[2])
    close(positive_score, 0.09634470710684997)
    close(negative_score, 0.8591327507278842)
    close(final_loss, 1.8584065774505372)
    print(f"SGNS：损失 {initial_loss:.9f} → {final_loss:.9f}")
    print(f"SGNS：三个梯度={analytic}")
    print(f"SGNS：正分数={positive_score:.9f}，负分数={negative_score:.9f}")
    print(f"SGNS：六个参数的有限差分最大误差={gradient_error:.3e}")
    close(softplus(1000.0), 1000.0)
    close(softplus(-1000.0), 0.0)
    close(sigmoid(-1000.0), 0.0)
    close(sigmoid(1000.0), 1.0)

    # Counts refer to generated word-context pairs. q(c)=N_c/N here.
    total, pair_count, word_count, context_count, negatives = 100, 10, 20, 25, 2
    q = context_count / total
    optimum = math.log(pair_count / (negatives * word_count * q))
    pmi = math.log(pair_count * total / (word_count * context_count))
    close(optimum, pmi - math.log(negatives))
    expected_derivative = (pair_count * (sigmoid(optimum) - 1.0)
                           + negatives * word_count * q * sigmoid(optimum))
    close(expected_derivative, 0.0)
    print(f"期望 SGNS：PMI={pmi:.9f}，PMI-log(K)={optimum:.9f}，导数为零")

    hidden = [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]
    pooled = masked_mean(hidden, [1, 1, 0])
    without_mask = masked_mean(hidden, [1, 1, 1])
    if pooled != [0.5, 0.5]:
        raise AssertionError("补齐位置混入了句向量")
    close(without_mask[0], 10.0 / 3.0)
    print(f"句向量：正确 masked mean={pooled}，漏 mask={without_mask}")

    similarities = [1.0, 0.0]
    loss, probabilities, similarity_gradients = contrastive_row(similarities, 0.5)
    close(loss, 0.1269280110429725)
    close(probabilities[0], 0.8807970779778823)
    close(similarity_gradients[0], -0.2384058440442351)
    close(similarity_gradients[1], 0.2384058440442351)
    for j, analytic_gradient in enumerate(similarity_gradients):
        plus, minus = list(similarities), list(similarities)
        plus[j] += 1e-6
        minus[j] -= 1e-6
        numerical_gradient = (contrastive_row(plus, 0.5)[0]
                              - contrastive_row(minus, 0.5)[0]) / 2e-6
        close(numerical_gradient, analytic_gradient, tolerance=1e-8)
    print(f"对比学习：正例概率={probabilities[0]:.9f}，单行损失={loss:.9f}")
    print(f"对比学习：缩放前相似度梯度={similarity_gradients}")

    query = [1.0, 0.0]
    documents = [[1.0, 0.0], [2.0, 2.0]]
    dots = [dot(query, document) for document in documents]
    cosines = [dot(normalize(query), normalize(document)) for document in documents]
    if not (dots[1] > dots[0] and cosines[0] > cosines[1]):
        raise AssertionError("未归一化的排名反例不成立")
    for document in documents:
        a, b = normalize(query), normalize(document)
        squared_distance = sum((x - y) ** 2 for x, y in zip(a, b))
        close(squared_distance, 2.0 - 2.0 * dot(a, b))
    print(f"检索：未归一化点积={dots}，余弦={cosines}，两种排名相反")
    for operation in (lambda: normalize([0.0, 0.0]),
                      lambda: masked_mean(hidden, [0, 0, 0]),
                      lambda: contrastive_row(similarities, 0.0)):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("无效数学输入未被拒绝")
    print("本章数值检查通过：只核对计算与一次 SGD 更新，未训练自然语言编码器。")


if __name__ == "__main__":
    main()
