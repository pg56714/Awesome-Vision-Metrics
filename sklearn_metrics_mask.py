from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
)
import numpy as np

test_cases = [
    (
        "完全重疊 (Complete Overlap)",
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
    ),
    (
        "部分重疊 (Partial Overlap)",
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]]),
    ),
    (
        "不重疊 (No Overlap)",
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]]),
    ),
    (
        "邊界接觸 (Touching at Edges)",
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]),
    ),
    (
        "小遮罩在大遮罩內 (Small Mask Inside Large Mask)",
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    ),
    (
        "交錯重疊 (Interleaved Overlap)",
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    ),
    (
        "不同形狀 (Different Shapes)",
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),
    ),
    (
        "相似形狀但位置偏移 (Similar Shapes but Offset)",
        np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]]),
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
    ),
    (
        "大面積交疊 (Large Area Overlap)",
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]]),
        np.array([[1, 1, 0], [1, 1, 1], [1, 0, 0]]),
    ),
    (
        "一個遮罩全為零 (One Mask All Zero)",
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1]]),
    ),
]

for description, true_mask, pred_mask in test_cases:
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()

    accuracy = accuracy_score(true_flat, pred_flat)
    precision = precision_score(true_flat, pred_flat, zero_division=0, average="binary")
    recall = recall_score(true_flat, pred_flat, zero_division=0, average="binary")
    f1 = f1_score(true_flat, pred_flat, zero_division=0, average="binary")
    mae = mean_absolute_error(true_flat, pred_flat)
    mse = mean_squared_error(true_flat, pred_flat)

    print(f"{description}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  Mean Absolute Error (MAE): {f1:.2f}")
    print(f"  Mean Squared Error (MSE): {f1:.2f}")
    print()
