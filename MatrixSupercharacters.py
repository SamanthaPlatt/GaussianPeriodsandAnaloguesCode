import numpy as np
import matplotlib as plt
import matplotlib.pyplot as pltfig
import sys
import itertools
from PIL import Image

#Computes the order of a matrix mod n
def computematrixorder(A,n):
    det = int(round(np.linalg.det(A)))
    g = np.gcd(n, det)
    if g > 1:
        print("A is not invertible mod n")
        sys.exit()

    size = len(A)
    j = 1
    A = np.matrix(A)
    B = A
    while not (B == np.identity(size)).all():
        B = np.array(B.dot(A))
        B %= n
        B = np.matrix(B)
        j = j + 1
    return j

#Computes A^b mod n for a matrix A
def matrixpowermod(A,b,n):
    m = 1
    B = np.matrix(A) % n
    if b == 0:
        return np.identity(len(B))
    else:
        while m < b:
            B = np.array(B.dot(A))
            B %= n
            B = np.matrix(B)
            m = m + 1
        return B

#m is the modulus, c is color modulus (currently set up so that the supercharacter values for two vectors
#v and w are given the same color if the sum of their coordinates are equivalent mod c),
#and A is an invertible matrix mod m. This is currently not as efficient as it could be --
#that is, some more optimization could be done to make the code run faster, especially when
#len(A) is bigger than 2
def MatrixSuperchar(A,n,c):
    hue = 1/(c+1)
    colors = []
    for f in range(1,c+1):
        g = f*hue
        colors.append(plt.colors.hsv_to_rgb((g, 0.7, (f+1)*hue)))
    A = np.array(A)
    A %= n
    n = len(A)
    A = np.matrix(A)
    I = np.identity(n)

    order = computematrixorder(A, n)
    print('n = '+str(n))
    print('order(A) = '+str(order))

    H = []
    B = I
    k = 0
    while k < order:
        H.append(B)
        B = B.dot(A)
        B %= n
        k = k + 1

    H_tuple = np.array(tuple(H), dtype=np.int64)
    plot = pltfig.gcf()
    ax = plot.add_subplot()
    # pltfig.title('modulus m = '+str(m)+'    matrix A = '+
    #              str(A)+'    order(A) = '+str(order)+'    c = '+str(c))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    theta = 2*np.pi/n
    ONE = [1] * len(A)
    iters = [range(n) for _ in range(len(A))]

    for e in range(c):
        reals = []
        imags = []
        prev_first = 0
        for variables in itertools.product(*iters):
            v = np.array(variables)
            l = v.sum() % c
            if l == e:
                C = (H_tuple.dot(ONE)).dot(v)
                C %= n
                C = C.astype(np.float64)
                C *= theta

                xs = np.cos(C)
                xsum = xs.sum(axis=0)

                ys = np.sin(C, out=xs)
                ysum = ys.sum(axis=0)

                reals.append(xsum)
                imags.append(ysum)

            if variables[0] != prev_first:
                prev_first = variables[0]
                ccc = '#' + ''.join(['%02x' % round(0xff * i) for i in colors[e]])
                ax.scatter(reals, imags, s=0.1, c=ccc)
                reals = []
                imags = []

        ccc = '#' + ''.join(['%02x' % round(0xff * i) for i in colors[e]])
        ax.scatter(reals, imags, s=1, c=ccc)

    pltfig.xlim([-order-0.1,order+0.1])
    pltfig.ylim([-order-0.1,order+0.1])
    plot.set_size_inches(12, 12)
    plot.savefig('MatrixSuperchar.png', dpi=240)

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        error = 64
        return (x0 - error, y0 - error, x1 + error, y1 + error)

    im = Image.open("MatrixSuperChar.png")
    im2 = im.crop(bbox(im))
    im2.save("MatrixSuperCharCropped.png")

import time
start_time = time.time()

n = 3003
c = 8
A = [[841, 351],
     [1911, 646]]

MatrixSuperchar(A,n,c)

end_time = time.time()
print(f"{end_time - start_time} seconds elapsed")