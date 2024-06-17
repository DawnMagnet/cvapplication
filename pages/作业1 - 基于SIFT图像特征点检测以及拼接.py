import math
import cv2
import numpy as np
from typing import List, Tuple, Union, Optional
import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
import io


def cv_show(name: str, image: np.ndarray, file_name: Optional[str] = None) -> None:
    """
    使用OpenCV显示图像，并可选择性地将其保存到文件。

    参数:
    - name (str): 用于显示图像的窗口名称。
    - image (numpy.ndarray): 要显示的图像，应为具有形状 (height, width, channels) 的 3D NumPy 数组。
    - file_name (str, 可选): 可选的文件路径，用于保存图像。如果不提供或为 None，则不会保存图像。

    返回:
    - None
    """
    cv2.imshow(name, image)
    cv2.waitKey()
    if file_name:
        cv2.imwrite(file_name, image)


def detectAndDescribe(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    检测图像中的关键点并计算其描述符。

    参数:
    - image (numpy.ndarray): 输入图像，应为具有形状 (height, width, channels) 的 3D NumPy 数组。

    返回:
    - kps (numpy.ndarray): 检测到的关键点数组，每个关键点是一个形状为 (x, y) 的数组。
    - features (numpy.ndarray): 对应的关键点描述符数组。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(gray, None)
    kps = np.float32([[kp.pt[0], kp.pt[1]] for kp in kps])
    return kps, features


def matchKeypoints(
    kpsA: np.ndarray,
    kpsB: np.ndarray,
    featuresA: np.ndarray,
    featuresB: np.ndarray,
    ratio: float = 0.75,
    reprojThresh: float = 4.0,
) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]]:
    """
    对两幅图像的关键点进行匹配，并计算匹配的单应性矩阵。

    参数:
    - kpsA (numpy.ndarray): 第一幅图像的关键点数组。
    - kpsB (numpy.ndarray): 第二幅图像的关键点数组。
    - featuresA (numpy.ndarray): 第一幅图像的关键点描述符数组。
    - featuresB (numpy.ndarray): 第二幅图像的关键点描述符数组。
    - ratio (float): 匹配比率，用于筛选好的匹配点。
    - reprojThresh (float): 重投影误差阈值，用于RANSAC算法。

    返回:
    - Optional[Tuple[List[Tuple[int, int]], numpy.ndarray, numpy.ndarray]]: 如果匹配成功，返回包含匹配点索引列表、单应性矩阵和状态数组的元组；否则返回 None。
    """
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return matches, H, status
    return None


def drawMatches(
    imageA: np.ndarray,
    imageB: np.ndarray,
    kpsA: np.ndarray,
    kpsB: np.ndarray,
    matches: List[Tuple[int, int]],
    status: np.ndarray,
) -> np.ndarray:
    """
    在两幅图像上绘制匹配的关键点，并连接它们。

    参数:
    - imageA (numpy.ndarray): 第一幅图像。
    - imageB (numpy.ndarray): 第二幅图像。
    - kpsA (numpy.ndarray): 第一幅图像的关键点数组。
    - kpsB (numpy.ndarray): 第二幅图像的关键点数组。
    - matches (List[Tuple[int, int]]): 匹配点的索引列表。
    - status (numpy.ndarray): 状态数组，指示哪些匹配点是好的。

    返回:
    - numpy.ndarray: 包含绘制的匹配线的图像。
    """
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    (_, _, p) = imageA.shape
    vis = np.zeros((max(hA, hB), wA + wB, p), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    for (trainIdx, queryIdx), s in zip(matches, status):
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    return vis


def stitch(
    imageA: np.ndarray,
    imageB: np.ndarray,
    M: Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray],
    ratio: float = 0.75,
    reprojThresh: float = 4.0,
    showMatches: bool = False,
):
    """
    拼接两幅图像。

    参数:
    - imageA (numpy.ndarray): 第一幅图像。
    - imageB (numpy.ndarray): 第二幅图像。
    - ratio (float): 匹配比率。
    - reprojThresh (float): 重投影误差阈值。
    - showMatches (bool): 是否显示匹配的关键点和连接线。

    返回:
    - Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None]: 返回拼接后的图像。如果 showMatches 为 True，则返回一个元组，包含拼接后的图像和带有匹配线的图像。
    """
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)
    (matches, H, status) = M
    result = cv2.warpPerspective(
        imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0])
    )
    r1 = result.copy()
    result[0 : imageB.shape[0], 0 : imageB.shape[1]] = imageB
    r2 = result.copy()
    if showMatches:
        vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        return result, vis
    return r1, r2


# num_points = st.slider("Number of points in spiral111", 1, 10000, 1100)
# num_turns = st.slider("Number of turns in spiral111", 1, 300, 31)

# indices = np.linspace(0, 1, num_points)
# theta = 2 * np.pi * num_turns * indices
# radius = indices

# x = radius * np.cos(theta)
# y = radius * np.sin(theta)

# df = pd.DataFrame(
#     {
#         "x": x,
#         "y": y,
#         "idx": indices,
#         "rand": np.random.randn(num_points),
#     }
# )

# st.altair_chart(
#     alt.Chart(df, height=700, width=700)
#     .mark_point(filled=True)
#     .encode(
#         x=alt.X("x", axis=None),
#         y=alt.Y("y", axis=None),
#         color=alt.Color("idx", legend=None, scale=alt.Scale()),
#         size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
#     )
# )

st.set_page_config(layout="wide")


"""
# 基于SIFT图像特征点检测以及拼接

SIFT算法由David Lowe于1999年提出，它是一种用于检测和描述图像中局部特征的方法。SIFT算法的主要步骤包括尺度空间极值检测、关键点定位、方向赋值和关键点描述。这些步骤共同确保了特征点的尺度不变性和旋转不变性，使得SIFT算法在图像拼接等应用中表现出色。
"""
col1, col2 = st.columns(2)
with col1:
    up1 = st.file_uploader(
        "上传第一个图片",
        type=["jpg", "png"],
        help="示例图片位置:examples/work1/left.png",
        accept_multiple_files=False,
    )
    if up1:
        st.image(up1.getvalue())

with col2:
    up2 = st.file_uploader(
        "上传第二个图片",
        type=["jpg", "png"],
        help="示例图片位置:examples/work1/right.png",
        accept_multiple_files=False,
    )
    if up2:
        st.image(up2.getvalue())


st.write(
    """
    ## 实验结果
         """
)
if up1 and up2:
    # 以下是代码中使用的图像读取和显示的示例
    # up1.
    a = Image.open(io.BytesIO(up1.getvalue()))
    b = Image.open(io.BytesIO(up2.getvalue()))
    imageA = cv2.resize(np.array(a), (600, 1000))
    imageB = cv2.resize(np.array(b), (600, 1000))
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)
    Match = matchKeypoints(kpsA, kpsB, featuresA, featuresB)
    if Match:
        (matches, H, status) = Match
        match = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        st.write("### 特征点匹配结果")
        st.image(match)
        p1, p2 = stitch(imageA, imageB, Match)
        st.write("### 图像拼接结果(右->左)")
        st.image(p1)
        st.write("### 图像拼接结果(左->右)")
        st.image(p2)
    else:
        st.write("匹配点数量不够")
else:
    st.write("上传图片后会自动进行计算，结果会显示在这里。")
