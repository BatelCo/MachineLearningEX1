# Batel Cohen
# 208521195

from init_centroids import init_centroids
from scipy.misc import imread
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys


# main function
def main_(path, k):
    loss = []
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    init = init_centroids(A, k)
    print("k=", k, ":")
    print("iter ", 0, ": ", print_cent_(init))
    min_dis_cen, dic = avg_(X, init)
    loss.append(get_loss_(X, init, min_dis_cen))
    for i in range(1, 11, 1):
        min_dis_cen, dic = avg_(X, init)
        update_(dic, init)
        print("iter ", i, ": ", print_cent_(init))
        loss.append(get_loss_(X, init, min_dis_cen))
    # update the points
    for j in range(len(min_dis_cen)):
        for i in range(k):
            if i == min_dis_cen[j]:
                X[j] = init[i]

    plt.imshow(X.reshape(A.shape))
    plt.grid(False)
    plt.show()
    return loss


# function that update centroids to be average of points
def update_(dictionary, init):
    average = 0
    for key in dictionary:
        average = np.mean(dictionary[key], axis=0)
        init[key] = average


# function that print the centroids
def print_cent_(centroid):
    if type(centroid) == list:
        centroid = np.asarray(centroid)
    if len(centroid.shape) != 1:
        return ' '.join(str(np.floor(100 * centroid) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(
            ' ]', ']').replace(' ', ', ')[1:-1]
    else:
        return ' '.join(str(np.floor(100 * centroid) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(
            ' ]', ']').replace(' ', ', ')


# function that find the most close centroid to given pixel
def find_minimum_(point, init):
    w = np.subtract(point, init[0])
    # the normal distance between centroid and pixel
    min = np.linalg.norm(w)
    min = pow(min, 2)
    min_cen = 0
    size = len(init)
    for i in range(1, size, 1):
        w = np.subtract(point, init[i])
        cur = np.linalg.norm(w)
        cur = pow(cur, 2)
        # minimum distance centroid setting to be the minimum
        if (cur < min):
            min = cur
            min_cen = i
            i += 1
    return min_cen


# function that find for each point the closest centroid
def avg_(points, init):
    dic = {}
    min_dis_cen = []
    i = 0
    min_centroid = 0
    for point in points:
        min_centroid = find_minimum_(point, init)
        min_dis_cen.append(min_centroid)
        i = i + 1
        if (min_centroid in dic):
            dic[min_centroid].append(point)
        else:
            arr = []
            arr.append(point)
            dic[min_centroid] = arr
    # 2)dictionary between the number of centroid and the closest pixels to it
    # min_dis_cen ia the list of most close centroids to the pixel
    return min_dis_cen, dic


# function that get loss of one iteration
def get_loss_(X, centroid, min_dis_cen):
    sum = 0
    for i in range(len(X)):
        distance = np.linalg.norm(X[i] - centroid[min_dis_cen[i]])
        sum = sum + distance
    return sum / len(X)


# calls main function
main = 'main_'
# relative path
path = "dog.jpeg"
lossesArray = []
k_num_array = [2, 4, 8, 16]
for k in k_num_array:
    lossesArray.append(main_(path, k))
print(lossesArray)
