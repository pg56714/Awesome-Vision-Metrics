import numpy as np


def calculate_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou


def calculate_mask_giou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    enclose_x_min = min(mask1_nonzero[0].min(), mask2_nonzero[0].min())
    enclose_x_max = max(mask1_nonzero[0].max(), mask2_nonzero[0].max())
    enclose_y_min = min(mask1_nonzero[1].min(), mask2_nonzero[1].min())
    enclose_y_max = max(mask1_nonzero[1].max(), mask2_nonzero[1].max())

    enclose_area = (enclose_x_max - enclose_x_min + 1) * (
        enclose_y_max - enclose_y_min + 1
    )

    giou = (intersection / union) - ((enclose_area - union) / enclose_area)
    return giou


def calculate_mask_diou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    mask1_center = np.mean(np.argwhere(mask1 == 1), axis=0)
    mask2_center = np.mean(np.argwhere(mask2 == 1), axis=0)

    distance = np.linalg.norm(mask1_center - mask2_center)

    enclose_x_min = min(mask1_nonzero[0].min(), mask2_nonzero[0].min())
    enclose_x_max = max(mask1_nonzero[0].max(), mask2_nonzero[0].max())
    enclose_y_min = min(mask1_nonzero[1].min(), mask2_nonzero[1].min())
    enclose_y_max = max(mask1_nonzero[1].max(), mask2_nonzero[1].max())

    c_diag = np.linalg.norm(
        np.array([enclose_x_max, enclose_y_max])
        - np.array([enclose_x_min, enclose_y_min])
    )

    diou = (intersection / union) - (distance**2 / c_diag**2)
    return diou


def calculate_mask_ciou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    mask1_center = np.mean(np.argwhere(mask1 == 1), axis=0)
    mask2_center = np.mean(np.argwhere(mask2 == 1), axis=0)
    distance = np.linalg.norm(mask1_center - mask2_center)
    enclose_x_min = min(mask1.nonzero()[0].min(), mask2.nonzero()[0].min())
    enclose_x_max = max(mask1.nonzero()[0].max(), mask2.nonzero()[0].max())
    enclose_y_min = min(mask1.nonzero()[1].min(), mask2.nonzero()[1].min())
    enclose_y_max = max(mask1.nonzero()[1].max(), mask2.nonzero()[1].max())
    c_diag = np.linalg.norm(
        np.array([enclose_x_max, enclose_y_max])
        - np.array([enclose_x_min, enclose_y_min])
    )
    mask1_shape = mask1.shape[0] / mask1.shape[1]
    mask2_shape = mask2.shape[0] / mask2.shape[1]
    v = (4 / np.pi**2) * (np.arctan(mask1_shape) - np.arctan(mask2_shape)) ** 2

    denominator = 1 - intersection / union + v
    if denominator != 0:
        alpha = v / denominator
    else:
        alpha = 0

    ciou = (intersection / union) - (distance**2 / c_diag**2 + alpha * v)
    return ciou


def calculate_mask_eiou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    mask1_center = np.mean(np.argwhere(mask1 == 1), axis=0)
    mask2_center = np.mean(np.argwhere(mask2 == 1), axis=0)
    distance = np.linalg.norm(mask1_center - mask2_center)
    enclose_x_min = min(mask1.nonzero()[0].min(), mask2.nonzero()[0].min())
    enclose_x_max = max(mask1.nonzero()[0].max(), mask2.nonzero()[0].max())
    enclose_y_min = min(mask1.nonzero()[1].min(), mask2.nonzero()[1].min())
    enclose_y_max = max(mask1.nonzero()[1].max(), mask2.nonzero()[1].max())
    c_diag = np.linalg.norm(
        np.array([enclose_x_max, enclose_y_max])
        - np.array([enclose_x_min, enclose_y_min])
    )
    eiou = (intersection / union) - (distance**2 / c_diag**2)
    return eiou


def calculate_focal_mask_eiou(mask1, mask2, gamma=2.0):
    eiou = calculate_mask_eiou(mask1, mask2)
    focal_eiou = (1 - eiou) ** gamma * eiou
    return focal_eiou


def calculate_mask_siou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    mask1_shape = mask1.shape[0] / mask1.shape[1]
    mask2_shape = mask2.shape[0] / mask2.shape[1]
    v = (4 / np.pi**2) * (np.arctan(mask1_shape) - np.arctan(mask2_shape)) ** 2
    siou = (intersection / union) - v
    return siou


def calculate_mask_alpha_iou(mask1, mask2, alpha=0.5):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    iou = intersection / union
    alpha_iou = iou**alpha
    return alpha_iou


def calculate_mask_wiou(mask1, mask2, weight=1):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    iou = intersection / union
    wiou = iou * weight
    return wiou


def calculate_mask_mpdiou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0

    # Check if either mask has no non-zero elements
    mask1_nonzero = mask1.nonzero()
    mask2_nonzero = mask2.nonzero()

    if mask1_nonzero[0].size == 0 or mask2_nonzero[0].size == 0:
        return 0.0

    mask1_center = np.mean(np.argwhere(mask1 == 1), axis=0)
    mask2_center = np.mean(np.argwhere(mask2 == 1), axis=0)
    distance = np.linalg.norm(mask1_center - mask2_center)
    enclose_x_min = min(mask1.nonzero()[0].min(), mask2.nonzero()[0].min())
    enclose_x_max = max(mask1.nonzero()[0].max(), mask2.nonzero()[0].max())
    enclose_y_min = min(mask1.nonzero()[1].min(), mask2.nonzero()[1].min())
    enclose_y_max = max(mask1.nonzero()[1].max(), mask2.nonzero()[1].max())
    c_diag = np.linalg.norm(
        np.array([enclose_x_max, enclose_y_max])
        - np.array([enclose_x_min, enclose_y_min])
    )
    mpdiou = (intersection / union) - (distance**2 / c_diag**2) - min(distance, c_diag)
    return mpdiou


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

for description, m1, m2 in test_cases:
    print(f"{description} - Mask IoU:", calculate_mask_iou(m1, m2))
    print(f"{description} - Mask GIoU:", calculate_mask_giou(m1, m2))
    print(f"{description} - Mask DIoU:", calculate_mask_diou(m1, m2))
    print(f"{description} - Mask CIoU:", calculate_mask_ciou(m1, m2))
    print(f"{description} - Mask EIoU:", calculate_mask_eiou(m1, m2))
    print(f"{description} - Mask Focal EIoU:", calculate_focal_mask_eiou(m1, m2))
    print(f"{description} - Mask SIoU:", calculate_mask_siou(m1, m2))
    print(f"{description} - Mask Alpha-IoU:", calculate_mask_alpha_iou(m1, m2))
    print(f"{description} - Mask WIoU:", calculate_mask_wiou(m1, m2))
    print(f"{description} - Mask MPDIoU:", calculate_mask_mpdiou(m1, m2))
    print()
