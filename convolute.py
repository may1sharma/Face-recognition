import matplotlib.pyplot as plt
import numpy as np


def convolute(image, filter):
    res = np.zeros((3,3))
    for i in range(5):
        for j in range(5):
            for m in range(i, i+3):
                for n in range(j, j+3):
                    if 1<m<5 and 1<n<5:
                        res[m-2,n-2] += image[i,j]*filter[m-i, n-j]
    print(res)
    plt.imshow(res, cmap='gray_r', interpolation='nearest')
    plt.savefig('conv.png')

image = np.array([[164, 188, 164, 161, 195],
                  [178, 201, 197, 150, 137],
                  [174, 168, 181, 190, 184],
                  [131, 179, 176, 185, 198],
                  [ 92, 185, 179, 133, 167]])
plt.imshow(image, cmap='gray_r', interpolation='nearest')
plt.savefig('image.png')
plt.clf()

filter = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])
convolute(image, filter)
plt.clf()

filter = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
convolute(image, filter)
plt.clf()

filter = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
convolute(image, filter)
plt.clf()