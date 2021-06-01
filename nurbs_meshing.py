'''
JasonChan.
2021 Apr.

'''

from Nurbsfun import *
import copy


def makeB8mesh(a, b, c, nnx, nny, nnz):
    '''
     Input:
     a,b,c: 三个方向的尺寸
     nnx: x方向的结点
    '''
    x1d = np.linspace(0, a, nnx)
    y1d = np.linspace(0, b, nny)
    z1d = np.linspace(0, c, nnz)

    count = 0
    node = np.zeros((nnx * nny * nnz, 3))
    for k in range(nnz):
        for j in range(nny):
            for i in range(nnx):
                node[count, :] = [x1d[i], y1d[j], z1d[k]]
                count = count + 1

    # build mesh
    chan = np.zeros((nnx, nnz, nny), int)
    count = 0
    for i in range(nnz):
        for j in range(nny):
            for k in range(nnx):
                chan[k, i, j] = count
                count = count + 1

    connecU = np.zeros((nnx - 1, 2), int)
    connecV = np.zeros((nny - 1, 2), int)
    connecW = np.zeros((nnz - 1, 2), int)

    for i in range(np.shape(connecU)[0]):
        connecU[i, :] = [i, i + 1]

    for i in range(np.shape(connecV)[0]):
        connecV[i, :] = [i, i + 1]

    for i in range(np.shape(connecW)[0]):
        connecW[i, :] = [i, i + 1]

    noElems = (nnx - 1) * (nny - 1) * (nnz - 1)
    element = np.zeros((noElems, 8), int)

    e = 0
    for w in range(nnz - 1):
        wConn = connecW[w, :]
        for v in range(nny - 1):
            vConn = connecV[v, :]
            for u in range(nnx - 1):
                c = 0
                uConn = connecU[u, :]
                for i in range(len(wConn)):
                    for j in range(len(vConn)):
                        for k in range(len(uConn)):
                            element[e, c] = chan[uConn[k], wConn[i], vConn[j]]
                            c = c + 1
                e = e + 1

    # renumbering nodes according to Jack Chessa's code
    col3 = element[:, 2].copy()
    col4 = element[:, 3].copy()
    col7 = element[:, 6].copy()
    col8 = element[:, 7].copy()

    element[:, 2] = col4
    element[:, 3] = col3
    element[:, 6] = col8
    element[:, 7] = col7

    return node, element


def SolidPoint(n, p, U, m, q, V, l, r, W, P, dim, u, v, w):
    '''
    参数坐标与物理坐标的转换
    INPUT:
     n         : number ob basis functions -1 !  - x-direction
            NURBS-Book: n+1 # basis, np max index (startindex 0)
            here        n   # basis and max index (startindex 1)
     p          : degree of the basis functions - x-direction
     U          : knotvector - x-direction
     m          : number ob basis functions -1 !  - y-direction
     q          : degree of the basis functions - y-direction
     V          : knotvector - y-direction
     P          : control points
     dim        : dimension of control points
     u          : xi-coordinate
     v          : eta-coordinate
     w          : zeta-coordinate
    OUTPUT:
     S          : coordinates of the point on the solid
    '''
    uspan = findspan(n, p, u, U)
    vspan = findspan(m, q, v, V)
    wspan = findspan(l, r, w, W)

    Nu = basisfuns(uspan, u, p, U)
    Nv = basisfuns(vspan, v, q, V)
    Nw = basisfuns(wspan, w, r, W)

    # compute point on solid using B-spline interpolation
    uind = uspan - p
    S = np.zeros((1, dim))

    for k in range(r + 1):
        wind = wspan - r + k
        for j in range(q + 1):
            vind = vspan - q + j
            for i in range(p + 1):
                CP = P[uind + i + (n + 1) * ((m + 1) * wind + vind), :]
                S = S + Nu[i] * Nv[j] * Nw[k] * CP
    return S


def nurb2proj(nob, locCP, locweights):
    '''
    %--------------------------------------------------------------
     transform NURBS data into projective coordinates
    INPUT:
     nob          : # of basis function = # control points / weights
     controlPoints: vector of control points (1 per row)
     weights :    : column vector of weights
    OUTPUT:
     projcoord    : matrix with projective coordinates
    --------------------------------------------------------------
    '''
    projcoord = copy.deepcopy(locCP)  #这里是个坑，要用到deepcopy否则原始数据会被修改 03-27
    for i in range(nob):
        projcoord[i, :] = projcoord[i, :] * locweights[i]

    locweights_T = np.mat(locweights).T
    projcoord_r = np.hstack((projcoord, locweights_T))
    return projcoord_r


def RefineKnotVectCurve(n, p, U, Pw, X, r):
    '''
    网格细化
    % NURBS-Book (algorithm A5.4) (modified)
    % insert multiple knots into curve
    %INPUT:
    % n         : number ob basis functions -1 !
    %        NURBS-Book: n+1 # basis, np max index (startindex 0)
    %        here        n   # basis and max index (startindex 1)
    % p          : degree of the basis functions
    % U         : old knotvector
    % Pw         : old control points
    % X          : vector of new knots (multiple entries possible)
    % r          :  (size of X) -1 (count the multple entries as well
    %             reason: same as above: max X index
    %OUTPUT:
    % Ubar        : newknot vector
    % Qw         : new control points
    '''

    dim = np.shape(Pw)[1]
    Qw = np.zeros((n + r + 2, dim))

    m = n + p + 1  # 3+2+1 = 6
    a = findspan(n, p, X[0], U)  # 2
    b = findspan(n, p, X[r], U)
    b = b + 1  # 4

    Ubar = np.zeros([1, m + r + 1 + 1])

    for j in range(0, a - p + 1):
        Qw[j] = Pw[j]
    for j in range(b - 1, n + 1):
        Qw[j + r + 1] = Pw[j]
    for j in range(0, a + 1):
        Ubar[0][j] = U[j]
    for j in range(b + p, m + 1):
        Ubar[0][j + r + 1] = U[j]  # 玄学，matlab会自动扩充数组长度，而python得先提前确定好数组长度。

    i = b + p - 1  # 4+2-1 = 5
    k = b + p + r  # 4+2+1 = 7

    for j in range(r, -1, -1):
        while (X[j] <= U[i] and i > a):
            Qw[k - p - 1] = Pw[i - p - 1]
            Ubar[0][k] = U[i]
            k = k - 1
            i = i - 1

        Qw[k - p - 1] = Qw[k - p]

        for l in range(1, p + 1):
            ind = k - p + l
            alfa = Ubar[0][k + l] - X[j]
            if (abs(alfa) == 0):
                Qw[ind - 1] = Qw[ind]
            else:
                alfa = alfa / (Ubar[0][k + l] - U[i - p + l])
                Qw[ind - 1] = alfa * Qw[ind - 1] + (1 - alfa) * Qw[ind]

        Ubar[0][k] = X[j]
        k = k - 1;

    return Ubar, Qw


def proj2nurbs(projcoord):
    '''
    %--------------------------------------------------------------
    % transform projective coordinates into NURBS data
    %INPUT:
    % projcoord    : matrix with projective coordinates
    %OUTPUT:
    % nob          : # of basis function = # control points / weights
    % controlPoints: vector of control points (1 per row)
    % weightVector : column vector of weights
    %--------------------------------------------------------------
    '''

    dimension = np.shape(projcoord)[1]
    weights = projcoord[:, dimension - 1]
    controlPoints = projcoord[:, 0:dimension - 1]

    for i in range(0, np.size(weights)):
        controlPoints[i, :] = controlPoints[i, :] * 1 / (weights[i])

    return controlPoints, weights
