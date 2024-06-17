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
# 作业5 - 欧式重构、仿射重构以及透视重构
## 实验背景
图像变换是计算机视觉领域的重要技术之一，可以用于图像增强、图像校正和对象识别等任务。在图像处理中，常用的变换包括欧式变换、仿射变换和透视变换。本实验旨在比较这三种变换在图像重构方面的效果。

## 实验目的
1. 比较欧式变换、仿射变换和透视变换对图像的影响。
2. 观察不同变换类型下图像的畸变情况。
3. 分析不同变换类型对图像信息的保留情况。

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

    def euclidean_reconstruction(image1, image2):
        # Perform Euclidean reconstruction (translation and rotation only)
        # Assume image1 and image2 are grayscale images of the same scene

        # Detect keypoints and compute descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Match keypoints between the two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Select top matches (adjust as needed)
        num_matches = 100
        matches = matches[:num_matches]

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Estimate rigid transformation (translation and rotation)
        transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Apply transformation to image1
        height, width = image1.shape[:2]
        image1_warped = cv2.warpAffine(image1, transformation_matrix, (width, height))

        return image1_warped

    def affine_reconstruction(image1, image2):
        # Perform Affine reconstruction (translation, rotation, scaling, and shearing)
        # Assume image1 and image2 are grayscale images of the same scene

        # Detect keypoints and compute descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Match keypoints between the two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Select top matches (adjust as needed)
        num_matches = 100
        matches = matches[:num_matches]

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Estimate affine transformation
        transformation_matrix = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

        # Apply transformation to image1
        height, width = image1.shape[:2]
        image1_warped = cv2.warpAffine(image1, transformation_matrix, (width, height))

        return image1_warped

    def perspective_reconstruction(image1, image2):
        # Perform perspective reconstruction
        # Assume image1 and image2 are grayscale images of the same scene

        # Detect keypoints and compute descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Match keypoints between the two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Select top matches (adjust as needed)
        num_matches = 100
        matches = matches[:num_matches]

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Estimate perspective transformation
        transformation_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply transformation to image1
        height, width = image1.shape[:2]
        image1_warped = cv2.warpPerspective(
            image1, transformation_matrix, (width, height)
        )

        return image1_warped

    # Perform reconstructions

    a = Image.open(io.BytesIO(up1.getvalue()))
    b = Image.open(io.BytesIO(up2.getvalue()))

    image1 = np.asarray(a)
    image2 = np.asarray(b)

    euclidean_result = euclidean_reconstruction(image1, image2)
    affine_result = affine_reconstruction(image1, image2)
    perspective_result = perspective_reconstruction(image1, image2)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("### 欧式重构")
        st.image(euclidean_result)
    with col2:
        st.write("### 仿射重构")
        st.image(affine_result)
    with col3:
        st.write("### 透视重构")
        st.image(perspective_result)
else:
    st.write("上传图片后会自动进行计算，结果会显示在这里。")


st.write(
    """
    ## 代码展示
    ```py
    def euclidean_reconstruction(image1, image2):
        # Perform Euclidean reconstruction (translation and rotation only)
        # Assume image1 and image2 are grayscale images of the same scene

        # Detect keypoints and compute descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Match keypoints between the two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Select top matches (adjust as needed)
        num_matches = 100
        matches = matches[:num_matches]

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Estimate rigid transformation (translation and rotation)
        transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Apply transformation to image1
        height, width = image1.shape[:2]
        image1_warped = cv2.warpAffine(image1, transformation_matrix, (width, height))

        return image1_warped

    def affine_reconstruction(image1, image2):
        # Perform Affine reconstruction (translation, rotation, scaling, and shearing)
        # Assume image1 and image2 are grayscale images of the same scene

        # Detect keypoints and compute descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Match keypoints between the two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Select top matches (adjust as needed)
        num_matches = 100
        matches = matches[:num_matches]

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Estimate affine transformation
        transformation_matrix = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

        # Apply transformation to image1
        height, width = image1.shape[:2]
        image1_warped = cv2.warpAffine(image1, transformation_matrix, (width, height))

        return image1_warped

    def perspective_reconstruction(image1, image2):
        # Perform perspective reconstruction
        # Assume image1 and image2 are grayscale images of the same scene

        # Detect keypoints and compute descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Match keypoints between the two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Select top matches (adjust as needed)
        num_matches = 100
        matches = matches[:num_matches]

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Estimate perspective transformation
        transformation_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply transformation to image1
        height, width = image1.shape[:2]
        image1_warped = cv2.warpPerspective(
            image1, transformation_matrix, (width, height)
        )

        return image1_warped
    ```
 """
)
