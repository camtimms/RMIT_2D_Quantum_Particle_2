# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:52:01 2023

@author: Campbell Timms
"""

import numpy as np
import matplotlib.pyplot as plt

# # Define the size of the checkerboard
# n = 8

# # Create the checkerboard pattern
# board = np.zeros((n, n))
# for i in range(n):
#     for j in range(n):
#         if (i + j) % 2 == 0:
#             board[i, j] = 1000

# # Plot the checkerboard as an image
# plt.imshow(board, cmap='gray')
# plt.show()

# Define the size of the checkerboard
n = 16

# Define the size of the square groups
group_size = 2

# Create the checkerboard pattern
board = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if (i // group_size + j // group_size) % 2 == 0:
            board[i, j] = 1000

# Plot the checkerboard as an image
plt.imshow(board, cmap='gray')
plt.show()
