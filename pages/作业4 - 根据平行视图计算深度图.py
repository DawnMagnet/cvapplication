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
# 作业4 - 根据平行视图计算深度图
## 要求：
进行根据平行视图计算深度图的实验时，通常包括了以下实验要求：
- 匹配点选取：选择匹配点的数量和分布适当，以确保能够准确地计算出深度图。通常需要足够多的匹配点，以覆盖整个场景或对象。特征匹配：使用适合的特征检测和匹配算法，如SIFT、SURF或ORB等，确保能够找到可靠的匹配点对。
- 透视变换计算：根据匹配点计算透视变换矩阵，将图像矫正为平行视图。确保透视变换过程准确，以避免深度图的误差。
- 深度计算：根据矫正后的图像进行深度计算。通常可以使用三角测量、立体视觉或其他深度估计方法来计算深度图。

## 算法概述：
1. 循环浏览像素
2. 区块匹配
```py
def ssd(left_block, right_block):
    # Calculate the Sum of Squared Differences (SSD) between two blocks
    return np.sum(np.square(np.subtract(left_block, right_block)))


def sad(left_block, right_block):
    # Calculate the Sum of Absolute Differences (SAD) between two blocks
    return np.sum(np.abs(np.subtract(left_block, right_block)))
```
3. 进行相似度计算
4. 寻找最佳差异
它根据计算出的相似度度量更新最佳差异和最佳相似性。最佳差异是使用 NCC 方法时产生最高相似度的差异，如果使用 SSD 或 SAD 时产生最小相似度的差异。
```py
def ncc(left_block, right_block):
    # Calculate the Normalized Cross-Correlation (NCC) between two blocks
    product = np.mean((left_block - left_block.mean()) * (right_block - right_block.mean()))
    stds = left_block.std() * right_block.std()

    if stds == 0:
        return 0
    else:
        return product / stds
```
5. 返回视差图

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

    def ncc(left_block, right_block):
        # Calculate the Normalized Cross-Correlation (NCC) between two blocks
        product = np.mean(
            (left_block - left_block.mean()) * (right_block - right_block.mean())
        )
        stds = left_block.std() * right_block.std()

        if stds == 0:
            return 0
        else:
            return product / stds

    def ssd(left_block, right_block):
        # Calculate the Sum of Squared Differences (SSD) between two blocks
        return np.sum(np.square(np.subtract(left_block, right_block)))

    def sad(left_block, right_block):
        # Calculate the Sum of Absolute Differences (SAD) between two blocks
        return np.sum(np.abs(np.subtract(left_block, right_block)))

    def select_similarity_function(method):
        # Select the similarity measure function based on the method name
        if method == "ncc":
            return ncc
        elif method == "ssd":
            return ssd
        elif method == "sad":
            return sad
        else:
            raise ValueError("Unknown method")

    def compute_disparity_map(
        bar, left_image, right_image, block_size, disparity_range, method="ncc"
    ):
        # Initialize disparity map
        height, width = left_image.shape[:2]
        disparity_map = np.zeros((height, width), np.uint8)
        half_block_size = block_size // 2
        similarity_function = select_similarity_function(method)

        # Loop over each pixel in the image
        all_times = height - 2 * half_block_size
        for row in range(half_block_size, height - half_block_size):
            bar.progress((row - half_block_size) / all_times, text="正在执行操作...")
            for col in range(half_block_size, width - half_block_size):
                best_disparity = 0
                best_similarity = (
                    float("inf") if method in ["ssd", "sad"] else float("-inf")
                )

                # Define one block for comparison based on the current pixel
                left_block = left_image[
                    row - half_block_size : row + half_block_size + 1,
                    col - half_block_size : col + half_block_size + 1,
                ]

                # Loop over different disparities
                for d in range(disparity_range):
                    if col - d < half_block_size:
                        continue

                    # Define the second block for comparison
                    right_block = right_image[
                        row - half_block_size : row + half_block_size + 1,
                        col - d - half_block_size : col - d + half_block_size + 1,
                    ]

                    # Compute the similarity measure
                    similarity = similarity_function(left_block, right_block)

                    # Update the best similarity and disparity if necessary
                    if method in ["ssd", "sad"]:
                        # For SSD and SAD, we are interested in the minimum value
                        if similarity < best_similarity:
                            best_similarity = similarity
                            best_disparity = d
                    else:
                        # For NCC, we are interested in the maximum value
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_disparity = d

                # Assign the best disparity to the disparity map
                disparity_map[row, col] = best_disparity * (256.0 / disparity_range)

        return disparity_map

    # 读取图像
    a = Image.open(io.BytesIO(up1.getvalue()))
    b = Image.open(io.BytesIO(up2.getvalue()))

    img1 = np.asarray(a)
    img2 = np.asarray(b)

    # Load images
    left_image, right_image = img1, img2
    block_size = 5
    disparity_range = 64  # This can be adjusted based on your specific context

    # Specify the similarity measurement method ('ncc', 'ssd', or 'sad')
    method = "ssd"  # Change this string to switch between methods

    # Compute the disparity map using the selected method

    progress_text = "正在执行操作..."
    my_bar = st.progress(0.0, text=progress_text)
    disparity_map = compute_disparity_map(
        my_bar, left_image, right_image, block_size, disparity_range, method=method
    )

    my_bar.empty()

    # Resize the disparity map for display
    scale_factor = 2.0  # Scaling the image by 3 times
    resized_image = cv2.resize(disparity_map, (0, 0), fx=scale_factor, fy=scale_factor)
    st.image(resized_image)
else:
    st.write("上传图片后会自动进行计算，结果会显示在这里。")


st.write(
    """
    ## 代码展示
    ```py
    def ncc(left_block, right_block):
        # Calculate the Normalized Cross-Correlation (NCC) between two blocks
        product = np.mean(
            (left_block - left_block.mean()) * (right_block - right_block.mean())
        )
        stds = left_block.std() * right_block.std()

        if stds == 0:
            return 0
        else:
            return product / stds

    def ssd(left_block, right_block):
        # Calculate the Sum of Squared Differences (SSD) between two blocks
        return np.sum(np.square(np.subtract(left_block, right_block)))

    def sad(left_block, right_block):
        # Calculate the Sum of Absolute Differences (SAD) between two blocks
        return np.sum(np.abs(np.subtract(left_block, right_block)))

    def select_similarity_function(method):
        # Select the similarity measure function based on the method name
        if method == "ncc":
            return ncc
        elif method == "ssd":
            return ssd
        elif method == "sad":
            return sad
        else:
            raise ValueError("Unknown method")

    def compute_disparity_map(
        bar, left_image, right_image, block_size, disparity_range, method="ncc"
    ):
        # Initialize disparity map
        height, width = left_image.shape[:2]
        disparity_map = np.zeros((height, width), np.uint8)
        half_block_size = block_size // 2
        similarity_function = select_similarity_function(method)

        # Loop over each pixel in the image
        all_times = height - 2 * half_block_size
        for row in range(half_block_size, height - half_block_size):
            bar.progress((row - half_block_size) / all_times, text="正在执行操作...")
            for col in range(half_block_size, width - half_block_size):
                best_disparity = 0
                best_similarity = (
                    float("inf") if method in ["ssd", "sad"] else float("-inf")
                )

                # Define one block for comparison based on the current pixel
                left_block = left_image[
                    row - half_block_size : row + half_block_size + 1,
                    col - half_block_size : col + half_block_size + 1,
                ]

                # Loop over different disparities
                for d in range(disparity_range):
                    if col - d < half_block_size:
                        continue

                    # Define the second block for comparison
                    right_block = right_image[
                        row - half_block_size : row + half_block_size + 1,
                        col - d - half_block_size : col - d + half_block_size + 1,
                    ]

                    # Compute the similarity measure
                    similarity = similarity_function(left_block, right_block)

                    # Update the best similarity and disparity if necessary
                    if method in ["ssd", "sad"]:
                        # For SSD and SAD, we are interested in the minimum value
                        if similarity < best_similarity:
                            best_similarity = similarity
                            best_disparity = d
                    else:
                        # For NCC, we are interested in the maximum value
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_disparity = d

                # Assign the best disparity to the disparity map
                disparity_map[row, col] = best_disparity * (256.0 / disparity_range)

        return disparity_map
    ```
 """
)
