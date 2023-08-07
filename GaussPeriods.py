import numpy as np
import matplotlib as plt
import matplotlib.pyplot as pltfig
import sys
from PIL import Image

#Computes the order of omega mod n
def OrderMod(omega,n):
    g = np.gcd(omega,n)
    if g > 1:
        print(str(omega)+" is not invertible mod "+str(n))
        sys.exit()

    j = 1
    b = omega
    while not b == 1:
        b = b*omega
        b %= n
        j = j + 1
    return j

#Generates the Gaussian period plot for modulus n, element omega, and color modulus c,
#where the Gaussian periods generated by two elements k and k' are given the same color if
#they are equivalent mod c. While the MatrixSupercharacters code does the same thing when
#inputting 1 by 1 matrices, but this version is much, much faster.
def GaussPeriodPlot(n, omega, c):
    hue = 1 / (c + 1)
    colors = []
    for a in range(1, c + 1):
        b = a * hue
        colors.append(plt.colors.hsv_to_rgb((b, 0.7, (a+1)*hue)))

    H = [1]
    i = omega
    j = 1
    while i != 1:
        H.append(i)
        i = i * omega % n
        j = j + 1

    H_tuple = np.array(tuple(H), dtype=np.int64)
    # print("H_tuple: ", H_tuple)
    plot = pltfig.gcf()
    ax = plot.add_subplot()
    # pltfig.title('n = '+str(n)+'    omega = '+str(omega)+'    d = '+str(j)+'    c = '+str(c))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    theta = 2 * np.pi / n

    print("n = "+str(n))
    print("order(omega) = "+str(len(H)))
    H_tuple.shape = (len(H_tuple), 1)

    step = 8 * 1024 * 1024 // 8
    for start in range(0, n, step):
        stop = min(start + step, n)

        for d in range(c):
            k = np.arange(start + d, stop, c)
            k = k * H_tuple
            k %= n
            k = k.astype(np.float64)
            k *= theta

            xs = np.cos(k)
            kxsum = xs.sum(axis=0)

            ys = np.sin(k, out=xs)
            kysum = ys.sum(axis=0)

            ccc = '#' + ''.join(['%02x' % round(0xff * i) for i in colors[d]])
            ax.scatter(kxsum, kysum, s=0.1, c=ccc)

    d = len(H)
    pltfig.xlim([-d - 0.1, d + 0.1])
    pltfig.ylim([-d - 0.1, d + 0.1])
    plot.set_size_inches(12, 12)
    plot.savefig('GaussPeriodPlot.png', dpi=240)

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        error = 64
        return (x0 - error, y0 - error, x1 + error, y1 + error)

    im = Image.open("GaussPeriodPlot.png")
    im2 = im.crop(bbox(im))
    im2.save("GaussPeriodPlotCropped.png")

import time
start_time = time.time()

n = 255255
omega = 254
c = np.gcd(omega - 1, n)
print('c = '+str(c))
GaussPeriodPlot(n,omega,c)

end_time = time.time()
print(f"{end_time - start_time} seconds elapsed")