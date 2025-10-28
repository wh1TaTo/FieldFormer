import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# 创建一个10x10的图像
img = np.zeros((10, 10))
img[4:7, 4:7] = 1

# 缩小图像
img_zoomed = zoom(img, 0.5)

# 显示原始图像和缩小后的图像
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(img_zoomed, cmap='gray')
plt.title('Zoomed Image')

plt.show()