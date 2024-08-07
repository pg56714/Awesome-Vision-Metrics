import numpy as np
from scipy.spatial.distance import directed_hausdorff


def pixel_accuracy(true_mask, pred_mask):
    """
    Compute the pixel accuracy between the true mask and the predicted mask.
    """
    assert (
        true_mask.shape == pred_mask.shape
    ), "Shape mismatch between true mask and predicted mask."
    correct_pixels = np.sum(true_mask == pred_mask)
    total_pixels = true_mask.size
    return correct_pixels / total_pixels


def dice_coefficient(true_mask, pred_mask):
    """
    Compute the Dice Coefficient between the true mask and the predicted mask.
    """
    assert (
        true_mask.shape == pred_mask.shape
    ), "Shape mismatch between true mask and predicted mask."
    intersection = np.sum((true_mask == 1) & (pred_mask == 1))
    union = np.sum(true_mask == 1) + np.sum(pred_mask == 1)
    dice = (2.0 * intersection) / union if union != 0 else 1.0
    return dice


def hausdorff_distance(true_mask, pred_mask):
    """
    Compute the Hausdorff distance between the true mask and the predicted mask.
    """
    true_points = np.argwhere(true_mask == 1)
    pred_points = np.argwhere(pred_mask == 1)

    if len(true_points) == 0 or len(pred_points) == 0:
        return float("inf")

    # Compute the directed Hausdorff distance
    forward_hausdorff = directed_hausdorff(true_points, pred_points)[0]
    backward_hausdorff = directed_hausdorff(pred_points, true_points)[0]

    return max(forward_hausdorff, backward_hausdorff)


# Test cases
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

# Run test cases
for description, true_mask, pred_mask in test_cases:
    print(f"Test Case: {description}")
    print("  Pixel Accuracy:", pixel_accuracy(true_mask, pred_mask))
    print("  Dice Coefficient:", dice_coefficient(true_mask, pred_mask))
    print("  Hausdorff Distance:", hausdorff_distance(true_mask, pred_mask))
    print()
