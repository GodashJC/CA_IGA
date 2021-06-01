'''
包含两个模型的参数
1.圆拱
2.梁

'''


from nurbs_meshing import *
from geometry_func import Geometry
from math import sqrt

#获取几何体数据
def arcdata(R, L, t, refineCount):
    Ri = R - t
    Rii = 0.5 * (R + Ri)
    hL = 0.5 * L

    E0 = 3e6  # Young's modulus
    nu0 = 0.3  # Poisson's ratio
    C = np.zeros((6, 6))

    C[0:3, :] = np.c_[E0 / (1 + nu0) / (1 - 2 * nu0) * np.array([[1 - nu0, nu0, nu0],
                                                                 [nu0, 1 - nu0, nu0],
                                                                 [nu0, nu0, 1 - nu0]]), np.zeros((3, 3))]

    C[3:6, :] = np.c_[np.zeros((3, 3)), E0 / (1 + nu0) * np.eye(3)]
    C = np.mat(C)

    controlPts = np.mat([[0, Ri, 0], [-Ri, Ri, 0], [-Ri, 0, 0], [-Ri, - Ri, 0], [0, - Ri, 0],
                         [0, Rii, 0], [-Rii, Rii, 0], [-Rii, 0, 0], [-Rii, - Rii, 0], [0, - Rii, 0],
                         [0, R, 0], [-R, R, 0], [-R, 0, 0], [-R, - R, 0], [0, - R, 0],
                         [0, Ri, hL], [-Ri, Ri, hL], [-Ri, 0, hL], [-Ri, - Ri, hL], [0, - Ri, hL],
                         [0, Rii, hL], [-Rii, Rii, hL], [-Rii, 0, hL], [-Rii, - Rii, hL], [0, - Rii, hL],
                         [0, R, hL], [-R, R, hL], [-R, 0, hL], [-R, - R, hL], [0, - R, hL],
                         [0, Ri, L], [-Ri, Ri, L], [-Ri, 0, L], [-Ri, - Ri, L], [0, - Ri, L],
                         [0, Rii, L], [-Rii, Rii, L], [-Rii, 0, L], [-Rii, - Rii, L], [0, - Rii, L],
                         [0, R, L], [-R, R, L], [-R, 0, L], [-R, - R, L], [0, - R, L]])

    p = 2
    q = 2
    r = 2

    uKnot = [0, 0, 0, 0.5, 0.5, 1, 1, 1]
    vKnot = [0, 0, 0, 1, 1, 1]
    wKnot = [0, 0, 0, 1, 1, 1]

    noPtsX = len(uKnot) - p - 1
    noPtsY = len(vKnot) - q - 1
    noPtsZ = len(wKnot) - r - 1
    # weights
    weights = np.ones((noPtsX * noPtsY * noPtsZ))
    fac = 1 / sqrt(2)

    weights_index = [2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 27, 29, 32, 34, 37, 39, 42, 44]
    for i in weights_index:
        weights[i - 1] = fac

    #  网格细化 h-refinement
    for c in range(refineCount):
        uKnotVectorU = np.unique(uKnot)
        uKnotVectorV = np.unique(vKnot)
        uKnotVectorW = np.unique(wKnot)

        # new knots along two directions
        newKnotsX = uKnotVectorU[0:-1] + 0.5 * np.diff(uKnotVectorU)
        newKnotsY = uKnotVectorV[0:-1] + 0.5 * np.diff(uKnotVectorV)
        newKnotsZ = uKnotVectorV[0:-1] + 0.5 * np.diff(uKnotVectorW)

        # h-refinement (NURBS) in x-direction
        dim = np.shape(controlPts)[1]  # 3
        nonewkX = np.size(newKnotsX)  # num of new knotX
        newprojcoord = np.zeros([noPtsX * noPtsY + nonewkX * noPtsY, dim + 1])

        rstart = 0
        wstart = 0

        for j in range(noPtsY):
            rstop = rstart + noPtsX - 1
            wstop = wstart + noPtsX - 1 + nonewkX

            # 注意numpy的切片操作 函数参数的深浅拷贝问题
            '''
            这个copy的问题是目前遇到的最大难题
            '''
            #         locCP = controlPts[rstart:rstop+1,:].copy()
            locCP = controlPts[rstart:rstop + 1, :]
            locweights = weights[rstart:rstop + 1]
            locprojcoord = nurb2proj(noPtsX, locCP, locweights)

            # print(locprojcoord)

            # refinement of x
            [tempknotVectorX, tempControlPts] = RefineKnotVectCurve(noPtsX - 1, p, uKnot, locprojcoord, newKnotsX,
                                                                    nonewkX - 1)
            newprojcoord[wstart:wstop + 1, :] = tempControlPts

            wstart = wstop + 1
            rstart = rstop + 1

        #     print(newprojcoord)

        uKnot = tempknotVectorX[0]  # 坑！！！ np.zeros定义的东西是 ([[]]); np.array是([])
        [controlPts, weights] = proj2nurbs(newprojcoord)
        noPtsX = noPtsX + nonewkX

    # refinement along r direction
    noElemsZ = 7
    knotWTemp = np.linspace(0, 1, noElemsZ + 2)
    wKnot = np.r_[[0, 0], knotWTemp, [1, 1]]
    noPtsZ = len(wKnot) - r - 1

    for i in range(noPtsZ - 1):
        a = (i + 1) * L / (noPtsZ - 1)
        controlPts = np.r_[controlPts,
                           np.c_[controlPts[0:noPtsX, 0:2], a * np.ones((noPtsX, 1))],
                           np.c_[controlPts[noPtsX:2 * noPtsX, 0:2], a * np.ones((noPtsX, 1))],
                           np.c_[controlPts[2 * noPtsX:3 * noPtsX, 0:2], a * np.ones((noPtsX, 1))],
        ]
        weights = np.r_[weights, weights[0:noPtsX], weights[0:noPtsX], weights[0:noPtsX]]

    geometry = Geometry(controlPts, uKnot, vKnot, wKnot, weights, p, q, r, noPtsX, noPtsY, noPtsZ, C)
    return geometry




def beam3dC1Data(a, b, c, noPtsX, noPtsY, noPtsZ, p, q, r):
    E0 = 1e5  # Young's modulus
    nu0 = 0.3  # Poisson's ratio
    # COMPUTE COMPLIANCE MATRIX
    C = np.zeros((6, 6))

    C[0:3, :] = np.c_[E0 / (1 + nu0) / (1 - 2 * nu0) * np.array([[1 - nu0, nu0, nu0],
                                                                 [nu0, 1 - nu0, nu0],
                                                                 [nu0, nu0, 1 - nu0]]), np.zeros((3, 3))]

    C[3:6, :] = np.c_[np.zeros((3, 3)), E0 / (1 + nu0) * np.eye(3)]
    C = np.mat(C)

    [controlPts, elementVV] = makeB8mesh(a, b, c, noPtsX, noPtsY, noPtsZ)

    # knot vectors
    knotUTemp = np.linspace(0, 1, noPtsX - p + 1)
    knotVTemp = np.linspace(0, 1, noPtsY - q + 1)
    knotWTemp = np.linspace(0, 1, noPtsZ - r + 1)
    uKnot = np.r_[[0, 0], knotUTemp, [1, 1]]
    vKnot = np.r_[[0, 0], knotVTemp, [1, 1]]
    wKnot = np.r_[[0, 0], knotWTemp, [1, 1]]

    # weights
    weights = np.ones((noPtsX * noPtsY * noPtsZ))

    geometry = Geometry(controlPts, uKnot, vKnot, wKnot, weights, p, q, r, noPtsX, noPtsY, noPtsZ, C)
    return geometry
