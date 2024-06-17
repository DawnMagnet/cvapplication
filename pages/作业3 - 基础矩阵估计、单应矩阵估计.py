import math
import cv2
import numpy as np
from typing import List, Tuple, Union, Optional
import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
import io

"""
# 基础矩阵估计、单应矩阵估计
## 要求：
根据所给的图像找出特征点对应关系（可以手动找，也可以用匹配算法找），根据归一化八点法求取基本矩阵，根据四点法求取单应矩阵。
 
"""
"""

## 算法概述：
1. 特征点检测和匹配：
在实验中，我们使用了SIFT特征检测器来检测图像中的关键点，并计算了每个关键点的特征描述子。然后使用FLANN匹配器对两幅图像的特征描述子进行匹配，筛选出最佳匹配的特征点。这些特征点将用于后续的基本矩阵和单应矩阵的计算。
2. 归一化八点法求取基本矩阵：
基于找到的特征点对，我们使用归一化八点法来估计基本矩阵F。该方法通过最小化八点法的误差来估计基本矩阵，这可以通过使用RANSAC算法来提高稳健性。得到的基本矩阵F表示了两个图像之间的几何关系，能够用于后续的三维重建或运动估计等任务。
3. 四点法求取单应矩阵：
利用找到的特征点对，我们使用四点法来估计单应矩阵H。单应矩阵是用于将一个图像中的点映射到另一个图像中的平面投影变换矩阵，通常用于图像配准和重叠。通过RANSAC算法，我们可以提高单应矩阵的稳健性，排除错误匹配的影响。

4. 对于基本矩阵F，我们得到了一个描述两个图像之间几何关系的矩阵。这个矩阵可以用于进行立体视觉中的相机姿态恢复或者三维重建等任务。
5. 对于单应矩阵H，我们得到了一个描述图像之间平面投影变换关系的矩阵。这个矩阵可以用于图像配准、景深合成、全景图像拼接等应用。
6. 归一化八点法和四点法是常用的图像配准和几何计算方法，可以有效地估计基本矩阵和单应矩阵。
7. RANSAC算法可以提高这些方法的稳健性，使其对于异常值和错误匹配具有更好的鲁棒性。
8. 基于特征点的图像配准方法可以应用于许多计算机视觉任务，如立体视觉、全景图像拼接、运动估计等。
通过实验结果分析，我们可以更好地理解图像配准方法的原理和应用，为后续的图像处理和计算机视觉任务提供了基础。

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

    def find_corresponding_points(img1, img2):
        # 使用SIFT特征检测器
        sift = cv2.SIFT_create()

        # 在图像中检测特征点和特征描述子
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # 使用FLANN匹配器进行特征点匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # 筛选出最佳匹配的特征点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # 提取出匹配的特征点的坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        return src_pts, dst_pts

    def find_fundamental_matrix(src_pts, dst_pts):
        # 使用归一化八点法求取基本矩阵
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)

        # 返回基本矩阵和对应的掩码
        return F, mask

    def find_homography_matrix(src_pts, dst_pts):
        # 使用四点法求取单应矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        # 返回单应矩阵和对应的掩码
        return H, mask

    # 读取图像
    img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

    # 找出特征点对应关系
    src_pts, dst_pts = find_corresponding_points(img1, img2)

    # 计算基本矩阵
    F, mask = find_fundamental_matrix(src_pts, dst_pts)

    # 计算单应矩阵
    H, mask = find_homography_matrix(src_pts, dst_pts)
    st.write("基本矩阵F：", F)
    st.write("单应矩阵H：", H)
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
