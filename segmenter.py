import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from skimage import data

np.set_printoptions(suppress=True)

class Segmenter:
    def __init__(self, n, L):
        self.n = n
        self.L = L

    def local_rel_entropy(self, image, x, y):
        width = image.shape[1]
        height = image.shape[0]

        i_lower = np.max([0, x - (self.n - 1) // 2])
        i_upper = np.min([x + (self.n - 1) // 2, width - 1]) + 1
        j_lower = np.max([0, y - (self.n - 1) // 2])
        j_upper = np.min([y + (self.n - 1) // 2, height - 1]) + 1

        mean_level = np.sum(image[j_lower:j_upper, i_lower:i_upper]) / self.n**2

        entropy = 0
        for i in range(i_lower, i_upper):
            for j in range(j_lower, j_upper):
                brightness = image[j, i]
                entropy +=  brightness * np.abs(np.log((brightness + 0.001) / mean_level))

        return entropy

    def get_brightness_entropy(self, image):
        width = image.shape[1]
        height = image.shape[0]

        values = np.ndarray([height, width, 2])

        for x in range(width):
            for y in range(height):
                values[y, x, 0] = image[y, x]
                values[y, x, 1] = self.local_rel_entropy(image, x, y)

        max_entropy = np.max(values[:, :, 1])
        min_entropy = np.min(values[:, :, 1])

        values[:, :, 1]  = np.floor((values[:, :, 1] - min_entropy) / (max_entropy - min_entropy) * (self.L - 1))

        return values

    def s_t_histogram(self, values):
        width = image.shape[1]
        height = image.shape[0]

        hist = np.zeros([self.L, self.L])

        for x in range(width):
            for y in range(height):
                b = values[y, x, 0]
                e = values[y, x, 1]
                hist[int(b), int(e)] += 1

        hist /= (width * height)

        return hist

    def mean_class_1(self, s, t, p):
        class_prob = np.sum(p[0:s, 0:t])

        I = np.arange(s).reshape(-1,1) + np.zeros((1, t))
        mu_1 = np.sum(I * p[0:s, 0:t]) / class_prob

        J = np.arange(t) + np.zeros((s, 1))
        mu_2 = np.sum(J * p[0:s, 0:t]) / class_prob

        return mu_1, mu_2

    def mean_class_2(self, s, t, p):
        class_prob = np.sum(p[s:, 0:t])

        I = np.arange(s, self.L).reshape(-1,1) + np.zeros((1, t))
        mu_1 = np.sum(I * p[s:self.L, 0:t]) / class_prob

        J = np.arange(t) + np.zeros((s, 1))
        mu_2 = np.sum(J * p[0:s, 0:t]) / class_prob

        return mu_1, mu_2

    def rel_entropy(self, s, t, p):
        mean_vector_1 = self.mean_class_1(s, t, p)
        mean_vector_2 = self.mean_class_2(s, t, p)

        entropy = 0

        I = np.arange(1, s).reshape(-1,1) + np.zeros((1, t - 1))
        entropy += np.sum(I * p[1:s, 1:t] * np.log(I / mean_vector_1[0]))

        J = np.arange(1, t) + np.zeros((s - 1, 1))
        entropy += np.sum(J * p[1:s, 1:t] * np.log(J / mean_vector_1[1]))

        I = np.arange(s, self.L).reshape(-1,1) + np.zeros((1, t - 1))
        entropy += np.sum(I * p[s:self.L, 1:t] * np.log(I / mean_vector_2[0]))

        J = np.arange(1, t) + np.zeros((self.L - s, 1))
        entropy += np.sum(J * p[s:self.L, 1:t] * np.log(J / mean_vector_2[1]))

        return entropy

    def segment(self, image):
        brightness_entropy = self.get_brightness_entropy(image)
        hist = self.s_t_histogram(brightness_entropy)

        ideal_threshold = -1
        min_entropy = np.inf
        for i in range(20, self.L):
            for j in range(20, self.L):
                entropy = self.rel_entropy(i, j, hist)
                if entropy < min_entropy:
                    ideal_threshold = (i, j)
                    min_entropy = entropy

        s_ideal, t_ideal = ideal_threshold
        print(ideal_threshold)
        print(min_entropy)

        mask = np.logical_and(brightness_entropy[:, :, 0] < 100, brightness_entropy[:, :, 1] < t_ideal)

        return mask

matplotlib.rcParams['font.size'] = 18

caller = getattr(data, 'coins')
image = caller()

seg = Segmenter(7, 256)
x = seg.segment(image)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(image, cmap=plt.cm.gray)
ax[1].imshow(x, cmap=plt.cm.gray)
plt.show()
