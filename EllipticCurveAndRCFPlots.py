import numpy as np
import cmath
import matplotlib as plt
import matplotlib.pyplot as pltfig
import sys
from PIL import Image

#Computes the Arithmetic-Geometric Mean using Definition 7.4.6 in Cohen
def AGM(x,y):
    error = 10**(-14)
    while abs(x-y) > error:
        x,y = (x + y)/2, np.sqrt(x*y)
    return x



#y^2 + a_1 xy + a_3 y = x^3 + a_2 x^2 + a_4 x + a_6
#from Cohen Algorithm 7.4.7: omega_1 is positive real, omega_2/omega_1 has
#positive imaginary part and real part 0 or -1/2
def lattice_basis(a_1, a_3, a_2, a_4, a_6):
    b_2 = a_1**2 + 4*a_2
    b_4 = a_1*a_3 + 2*a_4
    b_6 = a_3**2 + 4*a_6
    b_8 = (a_1**2)*a_6 + 4*a_2*a_6 - a_1*a_3*a_4 + a_2*(a_3**2) - a_4**2
    Delta = -(b_2**2)*b_8 - 8*(b_4**3) - 27*(b_6**2) + 9*b_2*b_4*b_6
    PolyCoeffs = [4,b_2,2*b_4,b_6]
    if Delta < 0:
        roots = np.roots(PolyCoeffs)
        for x in roots:
            if x.imag == 0:
                e_1 = x

        a = 3*e_1 + b_2/4
        b = np.sqrt(3*(e_1**2) + (b_2/2)*e_1 + b_4/2)
        agm_1 = AGM(2*np.sqrt(b),np.sqrt(2*b + a))
        agm_2 = AGM(2*np.sqrt(b),np.sqrt(2*b - a))
        omega_1 = 2*np.pi/agm_1
        omega_2 = complex(-omega_1/2, np.pi/agm_2)
    else:
        roots = np.roots(PolyCoeffs)
        roots = sorted(roots)
        e_3, e_2, e_1 = roots

        agm_1 = AGM(np.sqrt(e_1 - e_3),np.sqrt(e_1 - e_2))
        agm_2 = AGM(np.sqrt(e_1 - e_3),np.sqrt(e_2 - e_3))
        omega_1 = 2*np.pi/agm_1
        omega_2 = complex(0,np.pi/agm_2)
    return omega_1, omega_2



#Algorithm 7.4.2 in Cohen: uses lattice basis to find tau, and then finds
# matrix A in SL_2(Z) such that A*tau is in the fundamental domain
def fundamental_tau(omega_1, omega_2):
    tau = omega_2/omega_1
    A = np.array([[1,0],[0,1]])
    m = 0
    while m < 0.99999999:
        n = int(round(tau.real))
        tau = tau - n
        A = np.array([[1,-n],[0,1]]).dot(A)
        m = (abs(tau))**2
        if m < 0.99999999:
            tau = -1/tau
            A = np.array([[0,-1],[1,0]]).dot(A)
    return [tau, A]



#Combination of Cohen algorithms: Finds the coefficients g_2, g_3 such that the elliptic
#curve is y^2 = 4 x^3 - g_2 x - g_3
def WeierCoeffs(a_1,a_3,a_2,a_4,a_6):
    b_2 = a_1**2 + 4*a_2
    b_4 = a_1*a_3 + 2*a_4
    b_6 = a_3**2 + 4*a_6
    c_4 = b_2**2 - 24*b_4
    c_6 = -1*b_2**3 + 36*b_2*b_4 - 216*b_6
    g_2 = c_4/12
    g_3 = c_6/216
    return g_2, g_3

#Finds g_2 and g_3 for an elliptic curve y^2 = 4x^3 - g_2 x - g_3 that is isomorphic to
#C/lattice where the lattice has fundamental tau
def weiercoeffsgiventau(tau):
    i = complex(0,1)
    q = cmath.exp(2 * np.pi * i * tau)
    constantg2 = ((2 * np.pi)**4)/12
    constantg3 = ((2 * np.pi)**6)/216

    def q_expn_g2(n):
        q_to_n = q ** n
        computation = ((n ** 3) * q_to_n)/(1 - q_to_n)
        return computation

    def q_expn_g3(n):
        q_to_n = q ** n
        computation = ((n ** 5) * q_to_n)/(1 - q_to_n)
        return computation

    g2_comp = sum(q_expn_g2(n) for n in range(1, 20))
    g3_comp = sum(q_expn_g3(n) for n in range(1, 20))
    g_2 = constantg2*(1 + 240*g2_comp)
    g_3 = constantg3*(1 - 504*g3_comp)

    return g_2, g_3



#Algorithm 7.4.5 in Cohen: take lattice basis and a complex number z and
#output the corresponding point on the elliptic curve
def xweierstrass_p(omega_1, omega_2, z):
    tau = omega_2 / omega_1
    i = complex(0, 1)
    q = cmath.exp(2 * np.pi * i * tau)
    u = cmath.exp(2 * np.pi * i * z / omega_1)
    constant = (2 * np.pi * i / omega_1)
    xcomp = sum(
        (q ** n) * (u * (1 / ((1 - (q ** n) * u) ** 2) + 1 / (((q ** n) - u) ** 2)) - 2 / ((1 - (q ** n)) ** 2))
        for n in range(1, 15))
    x = (constant ** 2) * (1 / 12 + u / ((1 - u) ** 2) + xcomp)
    return x

def yweierstrass_p(omega_1, omega_2, z):
    tau = omega_2 / omega_1
    i = complex(0, 1)
    q = cmath.exp(2 * np.pi * i * tau)
    u = cmath.exp(2 * np.pi * i * z / omega_1)
    constant = (2 * np.pi * i / omega_1)
    ycomp = sum(
        (q ** n) * ((1 + (q ** n) * u) / ((1 - (q ** n) * u) ** 3) + ((q ** n) + u) / (((q ** n) - u) ** 3))
        for n in range(1, 15))
    y = (constant ** 3) * u * ((1 + u) / ((1 - u) ** 3) + ycomp)
    return y



#Same as previous, except inputs q = cmath.exp(2 * np.pi * i * tau) instead of
#a lattice basis, which works better for the plotting function later on
def xweierstrass_p_simplified(q, z):
    i = complex(0, 1)
    u = cmath.exp(2 * np.pi * i * z)
    constant = (2 * np.pi * i)
    def q_expn(n):
        q_to_n = q ** n
        computation = q_to_n * (u * (1 / ((1 - q_to_n * u) ** 2) + 1 / ((q_to_n - u) ** 2)) - 2 / ((1 - q_to_n) ** 2))
        return computation
    xcomp = sum(q_expn(n) for n in range(1, 15))
    x = (constant ** 2) * (1 / 12 + u / ((1 - u) ** 2) + xcomp)
    return x

def yweierstrass_p_simplified(q, z):
    i = complex(0, 1)
    u = cmath.exp(2 * np.pi * i * z)
    constant = (2 * np.pi * i)
    def q_expn(n):
        q_to_n = q ** n
        computation = q_to_n * ((1 + q_to_n * u) / ((1 - q_to_n * u) ** 3) + (q_to_n + u) / ((q_to_n - u) ** 3))
        return computation
    ycomp = sum(q_expn(n) for n in range(1, 15))
    y = (constant ** 3) * u * ( (1 + u)/((1 - u) ** 3) + ycomp)
    return y



#Finds and plots the x-coords (resp. y-coords) of the m-torsion points of the elliptic curve defined
#by y^2 + a_1 xy + a_3 y = x^3 + a_2 x^2 + a_4 x + a_6
def xtorsionplot(m, a_1, a_3, a_2, a_4, a_6):
    plot = pltfig.gcf()
    ax = plot.add_subplot()
    pltfig.title('The x-coords of the '+str(m)+'-torsion points for the elliptic curve y^2 + '+str(a_1)+'xy + '
                 +str(a_3)+'y = x^3 + '+str(a_2)+'x^2 + '+str(a_4)+'x + '+str(a_6))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    omega_1, omega_2 = lattice_basis(a_1, a_3, a_2, a_4, a_6)
    realavg = []
    imagavg = []

    for a in range(m):
        reals = []
        imags = []
        rng_start = 1 if a == 0 else 0
        for b in range(rng_start, m):
            z = (a/m)*omega_1 + (b/m)*omega_2
            x_complex = xweierstrass_p(omega_1,omega_2,z)
            reals.append(x_complex.real)
            imags.append(x_complex.imag)

        realavg.append(np.average(reals))
        imagavg.append(np.average(imags))
        ax.scatter(reals, imags, s=0.1)

    real = np.average(realavg)
    imag = np.average(imagavg)
    pltfig.xlim([real-100,real+100])
    pltfig.ylim([imag-100,imag+100])
    plot.set_size_inches(12, 12)
    plot.savefig('ECurveTorsionx.png', dpi=240)
    pltfig.show()

def ytorsionplot(m, a_1, a_3, a_2, a_4, a_6):
    plot = pltfig.gcf()
    ax = plot.add_subplot()
    pltfig.title('The y-coords of the ' + str(m) + '-torsion points for the elliptic curve y^2 + ' + str(a_1) + 'xy + '
                 + str(a_3) + 'y = x^3 + ' + str(a_2) + 'x^2 + ' + str(a_4) + 'x + ' + str(a_6))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    omega_1, omega_2 = lattice_basis(a_1, a_3, a_2, a_4, a_6)

    for a in range(m):
        reals = []
        imags = []
        rng_start = 1 if a == 0 else 0
        for b in range(rng_start, m):
            z = (a / m) * omega_1 + (b / m) * omega_2
            y_complex = yweierstrass_p(omega_1, omega_2, z)
            reals.append(y_complex.real)
            imags.append(y_complex.imag)

        ax.scatter(reals, imags, s=0.1)

    pltfig.xlim([-100, 100])
    pltfig.ylim([-10, 100])
    plot.set_size_inches(12, 12)
    plot.savefig('ECurveTorsiony.png', dpi=240)
    pltfig.show()



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



#Given a quadratic imaginary field K = \Q(\sqrt{-d}) and a matrix A of order "order" mod m,
#this returns a new "order" for A so that the subgroup generated by A avoids
#the group of units in the ring of integers O_K
def AvoidGroupofUnits(d,A,m,order):
    if d == 3:
        if order % 6 == 0:
            thirdroot = np.matrix([[0, -1], [1, -1]]) % m
            thirdrootsquared = matrixpowermod(thirdroot, 2, m)
            sixthroot = np.multiply(thirdroot, -1) % m
            othersixthroot = np.multiply(thirdrootsquared, -1) % m
            newmatrix = matrixpowermod(A, order // 6, m)
            if (newmatrix == sixthroot).all() or (newmatrix == othersixthroot).all():
                order = order // 6

        if order % 3 == 0:
            thirdroot = np.matrix([[0, -1], [1, -1]]) % m
            thirdrootsquared = matrixpowermod(thirdroot, 2, m)
            newmatrix = matrixpowermod(A, order // 3, m)
            if (newmatrix == thirdroot).all() or (newmatrix == thirdrootsquared).all():
                order = order // 3

    if d == 1:
        if order % 4 == 0:
            fourthroot = np.matrix([[0, -1], [1, 0]]) % m
            negfourthroot = np.multiply(fourthroot, -1) % m
            newmatrix = matrixpowermod(A, order // 4, m)
            if (newmatrix == fourthroot).all() or (newmatrix == negfourthroot).all():
                order = order // 4

    if order % 2 == 0:
        I = np.identity(2)
        negativeidentity = np.multiply(I, -1) % m
        if (matrixpowermod(A, order // 2, m) == negativeidentity).all():
            order = order // 2

    return order

#This is for the analogue of Gaussian sums for quadratic imaginary fields. In this case, d is the positive
#square-free integer such that K = \Q(\sqrt{-d}), m is the integer modulus, and a and b are chosen so that
#the matrix aI + bF is invertible mod m, where F is the companion matrix of f (the min poly for the ring
#integers). windowscale is a factor multiplier to affect size of axes in the final image, and should probably
#be between .01 and 1 in practice (use a bigger number when the modulus is bigger).
def RCFPPlot(d,m,a,b,windowscale):
    D = -d % 4
    I = np.identity(2)
    if D == 0:
        print("d must not be divisible by 4")
        sys.exit()
    if D == 1:
        companion = -((d+1)//4)
        F = [[0, companion],[1,-1]]
        omega = complex(-0.5, np.sqrt(d)/2)
    else:
        F = [[0,-d], [1,0]]
        omega = complex(0, np.sqrt(d))
    A = np.add(np.multiply(I,a), np.multiply(F,b))
    A %= m
    det = int(round(np.linalg.det(A)))
    g = np.gcd(m, det)
    if g > 1:
        print("aI + bF is not invertible mod m")
        sys.exit()

    print('m = ' + str(m))

    order = computematrixorder(A, m)
    print('order = '+str(order))

    order = AvoidGroupofUnits(d,A,m,order)
    print('new order = '+str(order))

    plot = pltfig.gcf()
    ax = plot.add_subplot()
    # pltfig.title('Discriminant D = '+str(d)+'     modulus m = '+str(m)+'     matrix A = '+
    #              str(a)+'*I + '+str(b)+'*F     order(A) = '+str(order))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    realavg = []
    imagavg = []
    i = complex(0,1)
    q = cmath.exp(2 * np.pi * i * omega)

    for x in range(m):
        reals = []
        imags = []
        rng_start = 1 if x == 0 else 0
        for y in range(rng_start, m):
            k = 0
            B = I
            terms_in_sum = []
            while k < order:
                v = [x,y]
                w = B.dot(v)
                w %= m
                z = (w[0]/m) + (w[1]/m)*omega
                term = xweierstrass_p_simplified(q,z)
                if d == 1:
                    term = term ** 2
                if d == 3:
                    term = term ** 3
                terms_in_sum.append(term)
                B = B.dot(A)
                B %= m
                k = k + 1
            sum_complex = sum(terms_in_sum)
            reals.append(sum_complex.real)
            imags.append(sum_complex.imag)

        realavg.append(np.average(reals))
        imagavg.append(np.average(imags))
        ax.scatter(reals, imags, s=0.1)

    real = np.average(realavg)
    imag = np.average(imagavg)
    maxavgwindow = max(abs(real),abs(imag))
    if d == 1:
        maxavgwindow = maxavgwindow ** (1/2)
    if d == 3:
        maxavgwindow = maxavgwindow ** (1/3)
    realstd = np.std(realavg)
    imagstd = np.std(imagavg)
    window = max(realstd,imagstd)
    if d == 1:
        window = window ** (1/2)
    if d == 3:
        window = window ** (1/3)
    pltfig.xlim([(-maxavgwindow-window)*windowscale,(maxavgwindow+window)*windowscale])
    pltfig.ylim([(-maxavgwindow-window)*windowscale,(maxavgwindow+window)*windowscale])
    plot.set_size_inches(12, 12)
    plot.savefig('RCFPPlot.png', dpi=240)

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        error = 64
        return (x0 - error, y0 - error, x1 + error, y1 + error)

    im = Image.open("RCFPPlot.png")
    im2 = im.crop(bbox(im))
    im2.save("RCFPPlotCropped.png")

#Same as previous, except magnitude of all points are mapped to the unit disc via
#the map |z| \mapsto |z|/(|z| + m^(1/root)). The root should be between 0 and 1, and a smaller root
#means the points close to 0 "explode out" more.
def RCFPPlotScaledToDisc(d,m,a,b,root):
    D = -d % 4
    I = np.identity(2)
    if D == 0:
        print("d must not be divisible by 4")
        sys.exit()
    if D == 1:
        companion = -((d+1)//4)
        F = [[0, companion],[1,-1]]
        omega = complex(-0.5, np.sqrt(d)/2)
    else:
        F = [[0,-d], [1,0]]
        omega = complex(0, np.sqrt(d))
    A = np.add(np.multiply(I,a), np.multiply(F,b))
    A %= m
    det = int(round(np.linalg.det(A)))
    g = np.gcd(m, det)
    if g > 1:
        print("aI + bF is not invertible mod m")
        sys.exit()

    print('m = ' + str(m))

    order = computematrixorder(A, m)
    print('order = '+str(order))

    order = AvoidGroupofUnits(d,A,m,order)
    print('new order = '+str(order))

    plot = pltfig.gcf()
    ax = plot.add_subplot()
    # pltfig.title('Discriminant d = '+str(d)+'     modulus m = '+str(m)+'     matrix A = '+
    #              str(a)+'*I + '+str(b)+'*F     order(A) = '+str(order))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    i = complex(0,1)
    q = cmath.exp(2 * np.pi * i * omega)

    for x in range(m):
        reals = []
        imags = []
        rng_start = 1 if x == 0 else 0
        for y in range(rng_start, m):
            k = 0
            B = I
            terms_in_sum = []
            while k < order:
                v = [x,y]
                w = B.dot(v)
                w %= m
                z = (w[0]/m) + (w[1]/m)*omega
                term = xweierstrass_p_simplified(q,z)
                if d == 1:
                    term = term ** 2
                if d == 3:
                    term = term ** 3
                terms_in_sum.append(term)
                B = B.dot(A)
                B %= m
                k = k + 1
            sum_complex = sum(terms_in_sum)
            mag = abs(sum_complex)
            scalefactor = m**(root)
            newmag = mag/(mag + scalefactor)
            phase = cmath.phase(sum_complex)
            zpoint = newmag*cmath.exp(phase*i)
            reals.append(zpoint.real)
            imags.append(zpoint.imag)

        ax.scatter(reals, imags, s=0.1)

    pltfig.xlim([-1,1])
    pltfig.ylim([-1,1])
    plot.set_size_inches(12, 12)
    plot.savefig('RCFPPlotScaledToDisc.png', dpi=240)

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        error = 64
        return (x0 - error, y0 - error, x1 + error, y1 + error)

    im = Image.open("RCFPPlotScaledToDisc.png")
    im2 = im.crop(bbox(im))
    im2.save("RCFPPlotScaledToDiscCropped.png")



#y-coords of torsion points of elliptic curve with CM by quad imgry field with discriminant d. windowscale
#works similarly to RCFPPlot.
def ECurveCMTorsiony(d,m,windowscale):
    D = -d % 4
    if D == 0:
        print("d must not be divisible by 4")
        sys.exit()
    if D == 1:
        omega = complex(-0.5, np.sqrt(d) / 2)
    else:
        omega = complex(0, np.sqrt(d))

    plot = pltfig.gcf()
    ax = plot.add_subplot()
    # pltfig.title('The y-coords of the '+str(m)+'-torsion points for the elliptic curve with CM by quadratic imaginary'
    #                                            ' field of discriminant = '+str(d))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    realavg = []
    imagavg = []
    i = complex(0, 1)
    q = cmath.exp(2 * np.pi * i * omega)

    for x in range(m):
        reals = []
        imags = []
        rng_start = 1 if x == 0 else 0
        for y in range(rng_start, m):
            z = (x / m) + (y / m) * omega
            term = yweierstrass_p_simplified(q, z)
            reals.append(term.real)
            imags.append(term.imag)

        realavg.append(np.average(reals))
        imagavg.append(np.average(imags))
        ax.scatter(reals, imags, s=0.1)

    real = np.average(realavg)
    imag = np.average(imagavg)
    realstd = np.std(realavg)
    imagstd = np.std(imagavg)
    window = max(realstd, imagstd)
    pltfig.xlim([real - windowscale * window, real + windowscale * window])
    pltfig.ylim([imag - windowscale * window, imag + windowscale * window])
    plot.set_size_inches(12, 12)
    plot.savefig('ECurveCMTorsiony.png', dpi=240)

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        error = 64
        return (x0 - error, y0 - error, x1 + error, y1 + error)

    im = Image.open("ECurveCMTorsiony.png")
    im2 = im.crop(bbox(im))
    im2.save("ECurveCMTorsionyCropped.png")

#Finds and plots the x-coords of the m-torsion points of the elliptic curve defined
#by y^2 + a_1 xy + a_3 y = x^3 + a_2 x^2 + a_4 x + a_6, where the dots are colored based
#their additive order. The window gives the min/max real and imaginary values on the axes.
def TorsionColorx(m,a_1,a_3,a_2,a_4,a_6, window):
    plot = pltfig.gcf()
    ax = plot.add_subplot()
    # pltfig.title('The x-coords of the '+str(m)+'-torsion points for the elliptic curve y^2 + '+str(a_1)+'xy + '
    #              +str(a_3)+'y = x^3 + '+str(a_2)+'x^2 + '+str(a_4)+'x + '+str(a_6))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    omega_1, omega_2 = lattice_basis(a_1, a_3, a_2, a_4, a_6)

    divisors = []
    for k in range(1,m+1):
        if m % k == 0:
            divisors.append(k)

    c = len(divisors)
    hue = 1 / (c + 1)
    colors = []

    for a in range(1, c + 1):
        b = a * hue
        colors.append(plt.colors.hsv_to_rgb((b, b, b)))

    for d in range(c):
        for a in range(m):
            reals = []
            imags = []
            rng_start = 1 if a == 0 else 0
            for b in range(rng_start, m):
                smallest = np.gcd(np.gcd(a,b), m)
                if smallest == divisors[d]:
                    z = (a/m)*omega_1 + (b/m)*omega_2
                    x_complex = xweierstrass_p(omega_1,omega_2,z)
                    reals.append(x_complex.real)
                    imags.append(x_complex.imag)

            ccc = '#' + ''.join(['%02x' % round(0xff * i) for i in colors[d]])
            ax.scatter(reals, imags, s=0.1, c = ccc)

    pltfig.xlim([-window, window])
    pltfig.ylim([-window, window])
    plot.set_size_inches(12, 12)
    plot.savefig('TorsionColorsx.png', dpi=240)

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        error = 64
        return (x0 - error, y0 - error, x1 + error, y1 + error)

    im = Image.open("TorsionColorsx.png")
    im2 = im.crop(bbox(im))
    im2.save("TorsionColorsxCropped.png")

#Finds and plots the x-coords of the m-torsion points of the elliptic curve which has CM by
#K = Q(sqrt{-d}) (where the lattice is chosen similarly to RCFPPlot), where the dots are sized (and colored)
#based on their additive order. The window gives the min/max real and imaginary values on the axes.
#The divisor helps control the size of the dots (larger divisor <--> smaller dots)
def TorsionDotSizex(m,d,window,divisor):
    D = -d % 4
    if D == 0:
        print("d must not be divisible by 4")
        sys.exit()
    if D == 1:
        omega = complex(-0.5, np.sqrt(d) / 2)
    else:
        omega = complex(0, np.sqrt(d))

    plot = pltfig.gcf()
    ax = plot.add_subplot()
    # pltfig.title('The x-coords of the '+str(m)+'-torsion points for the elliptic curve with CM by quadratic imaginary'
    #                                            ' field of discriminant = '+str(d))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    divisors = []

    for k in range(1,m+1):
        if m % k == 0:
            divisors.append(k)

    c = len(divisors)
    hue = 1 / (c + 1)
    colors = []

    for a in range(1, c + 1):
        b = a * hue
        colors.append(plt.colors.hsv_to_rgb((b, b, b)))

    for f in range(c):
        for a in range(m):
            div = divisors[f]
            reals = []
            imags = []
            rng_start = 1 if a == 0 else 0
            for b in range(rng_start,m):
                smallest = np.gcd(np.gcd(a,b), m)
                if smallest == div:
                    z = (a/m) + (b/m) * omega
                    x_complex = xweierstrass_p(1,omega,z)
                    reals.append(x_complex.real)
                    imags.append(x_complex.imag)

            order = m//div
            size = (m/(divisor*order))**2
            ccc = '#' + ''.join(['%02x' % round(0xff * i) for i in colors[f]])
            ax.scatter(reals, imags, s=size, c=ccc)

    pltfig.xlim([-window, window])
    pltfig.ylim([-window, window])
    plot.set_size_inches(12, 12)
    plot.savefig('TorsionDotsx.png', dpi=240)

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        error = 64
        return (x0 - error, y0 - error, x1 + error, y1 + error)

    im = Image.open("TorsionDotsx.png")
    im2 = im.crop(bbox(im))
    im2.save("TorsionDotsxCropped.png")

def TorsionDotSizey(m,d,window,divisor):
    D = -d % 4
    if D == 0:
        print("d must not be divisible by 4")
        sys.exit()
    if D == 1:
        omega = complex(-0.5, np.sqrt(d) / 2)
    else:
        omega = complex(0, np.sqrt(d))

    plot = pltfig.gcf()
    ax = plot.add_subplot()
    # pltfig.title('The y-coords of the '+str(m)+'-torsion points for the elliptic curve with CM by quadratic imaginary'
    #                                            ' field of discriminant = '+str(d))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_position('center')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    divisors = []

    for k in range(1,m+1):
        if m % k == 0:
            divisors.append(k)

    c = len(divisors)
    hue = 1 / (c + 1)
    colors = []

    for a in range(1, c + 1):
        b = a * hue
        colors.append(plt.colors.hsv_to_rgb((b, b, b)))

    for f in range(c):
        for a in range(m):
            div = divisors[f]
            reals = []
            imags = []
            rng_start = 1 if a == 0 else 0
            for b in range(rng_start,m):
                smallest = np.gcd(np.gcd(a,b), m)
                if smallest == div:
                    z = (a/m) + (b/m) * omega
                    y_complex = yweierstrass_p(1,omega,z)
                    reals.append(y_complex.real)
                    imags.append(y_complex.imag)

            order = m//div
            size = (m/(divisor*order))**2
            ccc = '#' + ''.join(['%02x' % round(0xff * i) for i in colors[f]])
            ax.scatter(reals, imags, s=size, c=ccc)

    pltfig.xlim([-window, window])
    pltfig.ylim([-window, window])
    plot.set_size_inches(12, 12)
    plot.savefig('TorsionDotsy.png', dpi=240)

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        error = 64
        return (x0 - error, y0 - error, x1 + error, y1 + error)

    im = Image.open("TorsionDotsy.png")
    im2 = im.crop(bbox(im))
    im2.save("TorsionDotsyCropped.png")



import time
start_time = time.time()

d = 7
m = 7**4
a = 1
b = 7**3
root = 1/2

#RCFPPlot(d,m,a,b,0.8)

#RCFPPlotScaledToDisc(d,m,a,b,root)

#ECurveCMTorsiony(d,m,0.03)

#TorsionDotSizey(m,d,70,10)

#TorsionDotSizey(m,d,70,10)

end_time = time.time()
print(f"{end_time - start_time} seconds elapsed")