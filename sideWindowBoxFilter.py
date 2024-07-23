import numpy as np
import matplotlib.pyplot as plt


class sdWinBoxFilt(object):

    def __init__(self, img, win, iter):
        self.image = img
        self.window = win
        self.iter = iter


    def generateFilterKernel(self):
        r = self.window
        k = np.ones((2*r+1, 1))/(2*r+1)
        k_L = k
        k_L[r+1:] = 0
        k_L = k_L / np.sum(k_L)  #行向量
        k_R = np.flipud(k_L)    #列向量
        self.k = k
        self.k_L = k_L
        self.k_R = k_R

    def convfilter(self, img, row_filter, col_filter):
        col_filter = np.transpose(col_filter)
        m, n = img.shape
        temp = img.copy()
        win = self.window

        for i in range(win, m-win-1):
            for j in range(win, n-win-1):
                block = img[i-win:i+win+1, j-win:j+win+1]
                temp[i, j] = col_filter @ (block @ row_filter)
        return temp - img


    def execute(self):
        img = self.image
        m, n, s = img.shape
        self.generateFilterKernel()

        k_L = self.k_L
        k_R = self.k_R
        k = self.k

        d = np.zeros((m, n, 8), np.float32)
        dm = np.zeros((m, n))
        img_out = np.zeros((m, n, s))
        for c in range(s):
            for i in range(self.iter):
                image = img[:, :, c]
                d[:, :, 0] = self.convfilter(image, k_L, k_L)
                d[:, :, 1] = self.convfilter(image, k_L, k_R)
                d[:, :, 2] = self.convfilter(image, k_R, k_L)
                d[:, :, 3] = self.convfilter(image, k_R, k_R)
                d[:, :, 4] = self.convfilter(image, k_L, k)
                d[:, :, 5] = self.convfilter(image, k_R, k)
                d[:, :, 6] = self.convfilter(image, k, k_L)
                d[:, :, 7] =  self.convfilter(image, k, k_R)

                TMP = np.abs(d)
                ind = np.argmin(TMP, 2)
                for y in range(m):
                    for x in range(n):
                        volum = d[y, x, :]
                        index = ind[y, x]
                        dm[y, x] = volum[index]

                image += dm
            img_out[:, :, c] = image
        return img_out


if __name__ == "__main__":

    I = plt.imread("SPYENV.png")
    plt.figure(1)
    plt.imshow(I)

    win = 3
    iter =1
    A = sdWinBoxFilt(I, win, iter)
    img_out = A.execute()
    plt.figure(2)
    plt.imshow(img_out)
    plt.title('deal-image')

    cmine = np.hstack((I, img_out))
    plt.figure(3)
    plt.imshow(cmine)
    plt.title('totalshow-image')

    plt.show()
