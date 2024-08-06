import numpy as np


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Check for zero area to avoid division by zero
    if box1_area == 0 or box2_area == 0:
        return 0.0

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def calculate_giou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Check for zero area to avoid division by zero
    if box1_area == 0 or box2_area == 0:
        return 0.0

    iou = inter_area / float(box1_area + box2_area - inter_area)
    x1_enclose = min(box1[0], box2[0])
    y1_enclose = min(box1[1], box2[1])
    x2_enclose = max(box1[2], box2[2])
    y2_enclose = max(box1[3], box2[3])
    enclose_area = (x2_enclose - x1_enclose) * (y2_enclose - y1_enclose)

    giou = iou - (enclose_area - (box1_area + box2_area - inter_area)) / enclose_area
    return giou


def calculate_diou(box1, box2):
    iou = calculate_iou(box1, box2)
    center_box1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center_box2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = np.linalg.norm(np.array(center_box1) - np.array(center_box2))
    x1_enclose = min(box1[0], box2[0])
    y1_enclose = min(box1[1], box2[1])
    x2_enclose = max(box1[2], box2[2])
    y2_enclose = max(box1[3], box2[3])
    c_diag = np.linalg.norm(
        np.array([x2_enclose, y2_enclose]) - np.array([x1_enclose, y1_enclose])
    )

    # Check for zero diagonal to avoid division by zero
    if c_diag == 0:
        return iou

    diou = iou - (distance**2 / c_diag**2)
    return diou


def calculate_ciou(box1, box2):
    iou = calculate_iou(box1, box2)
    center_box1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center_box2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = np.linalg.norm(np.array(center_box1) - np.array(center_box2))
    x1_enclose = min(box1[0], box2[0])
    y1_enclose = min(box1[1], box2[1])
    x2_enclose = max(box1[2], box2[2])
    y2_enclose = max(box1[3], box2[3])
    c_diag = np.linalg.norm(
        np.array([x2_enclose, y2_enclose]) - np.array([x1_enclose, y1_enclose])
    )

    width1 = box1[2] - box1[0]
    height1 = box1[3] - box1[1]
    width2 = box2[2] - box2[0]
    height2 = box2[3] - box2[1]

    # Check for zero height to avoid division by zero
    if height1 == 0 or height2 == 0 or c_diag == 0:
        return iou

    v = (4 / np.pi**2) * (
        np.arctan(width1 / height1) - np.arctan(width2 / height2)
    ) ** 2

    # Check if denominator is zero before computing alpha
    denominator = 1 - iou + v
    if denominator == 0:
        alpha = 0  # or choose a default value like alpha = 1
    else:
        alpha = v / denominator

    ciou = iou - (distance**2 / c_diag**2 + alpha * v)
    return ciou


def calculate_eiou(box1, box2):
    iou = calculate_iou(box1, box2)
    center_box1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center_box2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = np.linalg.norm(np.array(center_box1) - np.array(center_box2))
    x1_enclose = min(box1[0], box2[0])
    y1_enclose = min(box1[1], box2[1])
    x2_enclose = max(box1[2], box2[2])
    y2_enclose = max(box1[3], box2[3])
    c_diag = np.linalg.norm(
        np.array([x2_enclose, y2_enclose]) - np.array([x1_enclose, y1_enclose])
    )

    # Check for zero diagonal to avoid division by zero
    if c_diag == 0:
        return iou

    eiou = iou - (distance**2 / c_diag**2)
    return eiou


def calculate_focal_eiou(box1, box2, gamma=2.0):
    iou = calculate_iou(box1, box2)
    eiou = calculate_eiou(box1, box2)

    # Ensure Focal EIoU is 1 when completely overlapped
    if eiou == 1:
        return 1.0

    focal_eiou = (1 - eiou) ** gamma * eiou
    return focal_eiou


def calculate_siou(box1, box2):
    iou = calculate_iou(box1, box2)

    width1 = box1[2] - box1[0]
    height1 = box1[3] - box1[1]
    width2 = box2[2] - box2[0]
    height2 = box2[3] - box2[1]

    # Check for zero height to avoid division by zero
    if height1 == 0 or height2 == 0:
        return iou

    v = (4 / np.pi**2) * (
        np.arctan(width1 / height1) - np.arctan(width2 / height2)
    ) ** 2
    siou = iou - v
    return siou


def calculate_alpha_iou(box1, box2, alpha=0.5):
    iou = calculate_iou(box1, box2)
    alpha_iou = iou**alpha
    return alpha_iou


def calculate_wiou(box1, box2, weight=1):
    iou = calculate_iou(box1, box2)
    wiou = iou * weight
    return wiou


def calculate_mpdiou(box1, box2):
    iou = calculate_iou(box1, box2)
    center_box1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center_box2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = np.linalg.norm(np.array(center_box1) - np.array(center_box2))
    x1_enclose = min(box1[0], box2[0])
    y1_enclose = min(box1[1], box2[1])
    x2_enclose = max(box1[2], box2[2])
    y2_enclose = max(box1[3], box2[3])
    c_diag = np.linalg.norm(
        np.array([x2_enclose, y2_enclose]) - np.array([x1_enclose, y1_enclose])
    )

    # Check for zero diagonal to avoid division by zero
    if c_diag == 0:
        return iou

    mpdiou = iou - (distance**2 / c_diag**2) - min(distance, c_diag)
    return mpdiou


# Test cases
test_cases = [
    ("完全重疊 (Complete Overlap)", [0, 0, 2, 2], [0, 0, 2, 2]),
    ("部分重疊 (Partial Overlap)", [0, 0, 2, 2], [1, 1, 3, 3]),
    ("不重疊 (No Overlap)", [0, 0, 2, 2], [3, 3, 5, 5]),
    ("邊界接觸 (Touching at Edges)", [0, 0, 2, 2], [2, 2, 4, 4]),
    ("小框在大框內 (Small Box Inside Large Box)", [1, 1, 2, 2], [0, 0, 3, 3]),
    ("交錯重疊 (Interleaved Overlap)", [0, 0, 3, 3], [1, 1, 4, 4]),
    ("不同形狀 (Different Shapes)", [0, 0, 2, 3], [1, 0, 3, 2]),
    (
        "相似形狀但位置偏移 (Similar Shapes but Offset)",
        [0, 0, 2, 2],
        [0.5, 0.5, 2.5, 2.5],
    ),
    ("大面積交疊 (Large Area Overlap)", [0, 0, 4, 4], [1, 1, 3, 3]),
    ("一個框全為零 (One Box All Zero)", [0, 0, 0, 0], [1, 1, 2, 2]),
]

for description, b1, b2 in test_cases:
    print(f"{description} - IoU:", calculate_iou(b1, b2))
    print(f"{description} - GIoU:", calculate_giou(b1, b2))
    print(f"{description} - DIoU:", calculate_diou(b1, b2))
    print(f"{description} - CIoU:", calculate_ciou(b1, b2))
    print(f"{description} - EIoU:", calculate_eiou(b1, b2))
    print(f"{description} - Focal EIoU:", calculate_focal_eiou(b1, b2))
    print(f"{description} - SIoU:", calculate_siou(b1, b2))
    print(f"{description} - Alpha-IoU:", calculate_alpha_iou(b1, b2))
    print(f"{description} - WIoU:", calculate_wiou(b1, b2))
    print(f"{description} - MPDIoU:", calculate_mpdiou(b1, b2))
    print()
