from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import cv2


class cannyEdgeDetector:
    def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05,
                 highthreshold=0.15):
        self.img = img
        self.img_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return g

    def non_max_supression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # Angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # Angle 45
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # Angle 90
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # Angle 135
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]
                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
        return Z

    def hysteresis(self, img):
        img = img[0]
        M, N = img.shape

        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img[i, j] == weak:
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or
                                (img[i + 1, j + 1] == strong) or (img[i, j - 1] == strong) or
                                (img[i, j + 1] == strong) or (img[i - 1, j - 1] == strong) or
                                (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img

    def sobel_filters(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)

        return G, theta

    def threshold(self, img):
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res, weak, strong

    def my_normalize(self, img):
        if len(img.shape) == 3:  # check if img is color
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # RGB to grayscale

        # convert into range of [0,1]
        min_val = np.min(img.ravel())
        max_val = np.max(img.ravel())
        output = (img.astype('float') - min_val) / (max_val - min_val)

        return output

    def detect(self):
        self.img = self.my_normalize(self.img)
        self.img_smoothed = convolve(self.img, self.gaussian_kernel(self.kernel_size, self.sigma))
        self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
        self.nonMaxImg = self.non_max_supression(self.gradientMat, self.thetaMat)
        self.thresholdImg = self.threshold(self.nonMaxImg)
        img_final = self.hysteresis(self.thresholdImg)
        self.img_final = img_final

        return self.img_final


# Load Image
image_name = 'rubestiti.jpg'
img = cv2.imread(image_name)

# Canny Edge Detection
canny_edge = cannyEdgeDetector(img)
canny_edge_detection = canny_edge.detect()

# Plot Detection
plt.imshow(canny_edge_detection, cmap='gray')
plt.show()
