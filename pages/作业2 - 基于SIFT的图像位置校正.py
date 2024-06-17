import math
import cv2
import numpy as np
from typing import List, Tuple, Union, Optional
import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
import io

st.set_page_config(layout="wide")

"""
# 作业2 - 基于SIFT的图像位置校正
## 要求：
将一对图像校正为平行视图，令两幅图像“平行”。如图所示：
 
"""

st.image(Image.open("pages/w2-1.png"))
"""

## 算法概述：
1. 特征提取和匹配： 使用特征检测算法（如SIFT、SURF、ORB等）在两幅图像中提取特征点，并通过特征描述子进行匹配。本次实验采用SIFT检测算法，特征检测步骤：a、搜索所有尺度空间上的图像，通过高斯微分函数来识别潜在的对尺度和选择不变的兴趣点。b、通过拟合精细模型来确定位置尺度，关键点的选取依据他们的稳定程度。c、基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向，以便后续对关键点的方向、尺度和位置进行变换操作。d、在每个特征点周围的邻域内，在选定的尺度上测量图像的局部梯度。

2. 计算基础矩阵及极点：基础矩阵F是3x3矩阵，表达了立体像对的像点之间的对应关系。对于每个图像中的特征点，基础矩阵定义了在另一幅图像上的对应点所在的极线。通过基础矩阵和对应点的坐标，可以计算出在另一幅图像中的极线。使用选定的对应点，可以利用八点法或基于迭代优化的方法（如RANSAC）来估计基础矩阵。根据基础矩阵或基本矩阵，计算两个视图中对应特征点的极线。极线是通过一个视图中的点与另一个视图中的对应点确定的直线。


3. 校正变换计算：校正变换是用于将两幅图像校正到一个共同平面上的变换，以使它们的特定特征或结构保持平行或对齐。在立体视觉中，校正变换通常用于校正成对的图像，使它们具有相同的视角和几何形状。

4. 图像校正：应用校正变换将两幅图像进行校正，使得它们在校正后保持平行。

## 实验环境

- Python=3.9
- Numpy
- OpenCV=4.4.0.46

## 实验结果

"""

col1, col2 = st.columns(2)
with col1:
    up1 = st.file_uploader(
        "上传第一个图片",
        type=["jpg", "png"],
        help="示例图片位置:examples/work2/left.png",
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

    def rectify_images(img1, img2):
        # 1. 特征匹配
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # 2. 应用比率测试以选择良好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 提取匹配的关键点
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # 3. 计算基础矩阵
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 0.1, 0.99)

        # 4. 计算极线
        epilines1 = cv2.computeCorrespondEpilines(dst_pts, 2, F)
        epilines1 = epilines1.reshape(-1, 3)
        epilines2 = cv2.computeCorrespondEpilines(src_pts, 1, F)
        epilines2 = epilines2.reshape(-1, 3)

        # 5. 校正变换
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        _, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, (w1, h1))

        # 6. 校正图像
        img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
        img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

        return img1_rectified, img2_rectified

    # 读取图像

    a = Image.open(io.BytesIO(up1.getvalue()))
    b = Image.open(io.BytesIO(up2.getvalue()))

    img1 = np.asarray(a)
    img2 = np.asarray(b)

    # 图像校正
    img1_rectified, img2_rectified = rectify_images(img1, img2)
    st.write("### 图像拼接结果(右->左)")
    st.image(img1_rectified)
    st.write("### 图像拼接结果(左->右)")
    st.image(img2_rectified)
else:
    st.write("上传图片后会自动进行计算，结果会显示在这里。")


st.write(
    """
    ## 代码展示
    ```py
    def rectify_images(img1, img2):
        # 1. 特征匹配
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # 2. 应用比率测试以选择良好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 提取匹配的关键点
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # 3. 计算基础矩阵
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 0.1, 0.99)

        # 4. 计算极线
        epilines1 = cv2.computeCorrespondEpilines(dst_pts, 2, F)
        epilines1 = epilines1.reshape(-1, 3)
        epilines2 = cv2.computeCorrespondEpilines(src_pts, 1, F)
        epilines2 = epilines2.reshape(-1, 3)

        # 5. 校正变换
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        _, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, (w1, h1))

        # 6. 校正图像
        img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
        img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

        return img1_rectified, img2_rectified
    ```
 """
)
