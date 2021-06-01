from Nurbsfun import *
from liner_elastic import strainDispMatrix3d
'''
等几何分析函数包
2021 Apr. 
JasonChan.
HNU.

'''

def buildConnectivity(p, knotVec, noElems):
    '''
    对每个维度进行划分网格
    '''
    elRange = np.zeros([noElems, 2])
    elKnotIndices = np.zeros([noElems, 2])
    elConn = np.zeros([noElems, p + 1], int)

    element = 0
    previousKnotVal = 0

    for i in range(len(knotVec)):
        currentKnotVal = knotVec[i]
        if knotVec[i] != previousKnotVal:
            elRange[element, :] = [previousKnotVal, currentKnotVal]
            elKnotIndices[element, :] = [i - 1, i]
            element = element + 1
        previousKnotVal = currentKnotVal

    for e in range(noElems):
        elConn[e, :] = [i for i in range(int(elKnotIndices[e, 0] - p), int(elKnotIndices[e, 0] + 1))]

    return elRange, elConn


# 生成IGA3D网格
def generateIGA3DMesh(geometry):
    uniqueUKnots = np.unique(geometry.uKnot)
    uniqueVKnots = np.unique(geometry.vKnot)
    uniqueWKnots = np.unique(geometry.wKnot)
    noElemsU = len(uniqueUKnots) - 1
    noElemsV = len(uniqueVKnots) - 1
    noElemsW = len(uniqueWKnots) - 1

    chan = np.zeros([geometry.noPtsX, geometry.noPtsZ, geometry.noPtsX])

    count = 0
    for i in range(geometry.noPtsZ):
        for j in range(geometry.noPtsY):
            for k in range(geometry.noPtsX):
                chan[k, i, j] = count
                count += 1

    [elRangeU, elConnU] = buildConnectivity(geometry.p, geometry.uKnot, noElemsU)
    [elRangeV, elConnV] = buildConnectivity(geometry.q, geometry.vKnot, noElemsV)
    [elRangeW, elConnW] = buildConnectivity(geometry.r, geometry.wKnot, noElemsW)

    noElems = noElemsU * noElemsV * noElemsW
    element = np.zeros([noElems, (geometry.p + 1) * (geometry.q + 1) * (geometry.r + 1)], int)

    e = 0
    for w in elConnW:
        for v in elConnV:
            for u in elConnU:
                c = 0
                for i in range(len(w)):
                    for j in range(len(v)):
                        for k in range(len(u)):
                            element[e, c] = chan[u[k], w[i], v[j]]
                            c += 1
                e += 1

    index = np.zeros([noElems, 3], int)
    count = 0
    for i in range(len(elRangeW)):
        for j in range(len(elRangeV)):
            for k in range(len(elRangeU)):
                index[count, :] = [k, j, i]
                count += 1

    setattr(geometry, "index", index)
    setattr(geometry, "element", element)
    setattr(geometry, "elRangeU", elRangeU)
    setattr(geometry, "elRangeV", elRangeV)
    setattr(geometry, "elRangeW", elRangeW)
    setattr(geometry, "noElems", noElems)

#坐标转换以及雅可比矩阵
def parent2parametric(xiE, pt):
    xi = 0.5 * ((xiE[1] - xiE[0]) * pt + xiE[1] + xiE[0])
    return xi


# from parent to parametric space
def jacobianPaPaMapping(rangeU, rangeV):
    return 0.5 * (rangeU[1] - rangeU[0]) * 0.5 * (rangeV[1] - rangeV[0])


def jacobianPaPaMapping3d(rangeU, rangeV, rangeW):
    J2xi = 0.5 * (rangeU[1] - rangeU[0])
    J2eta = 0.5 * (rangeV[1] - rangeV[0])
    J2zeta = 0.5 * (rangeW[1] - rangeW[0])
    j = J2xi * J2eta * J2zeta
    return j


def assembly_K(geometry, Q, W):
    # 组装整体刚度矩阵以及载荷向量
    K = np.mat(np.zeros((geometry.noDofs, geometry.noDofs)))

    index = getattr(geometry, 'index')
    element = getattr(geometry, 'element')
    elRangeU = getattr(geometry, 'elRangeU')
    elRangeV = getattr(geometry, 'elRangeV')
    elRangeW = getattr(geometry, 'elRangeW')
    noElems = getattr(geometry, 'noElems')

    # 对每个单元进行循环
    for e in range(noElems):
        idu = index[e, 0]
        idv = index[e, 1]
        idw = index[e, 2]
        xiE = elRangeU[idu]  # [xi_i,xi_i+1]
        etaE = elRangeV[idv]
        zetaE = elRangeW[idw]

        sctr = element[e]  # IEN array 连接数组
        sctrB = np.hstack(
            (sctr, sctr + geometry.noCtrPts, sctr + 2 * geometry.noCtrPts))  # vector that scatters a B matrix
        nn = len(sctr)

        # 循环高斯点
        for gp in range(len(W)):
            pt = Q[gp]
            wt = W[gp]

            # 坐标变换 parent to parametric t
            Xi = parent2parametric(xiE, pt[0])
            Eta = parent2parametric(etaE, pt[1])
            Zeta = parent2parametric(zetaE, pt[2])
            J2 = jacobianPaPaMapping3d(xiE, etaE, zetaE)

            # 计算参数坐标上的基函数微分
            [N, dRdxi, dRdeta, dRdzeta] = NURBS3DBasisDers(Xi, Eta, Zeta, geometry.p, geometry.q, geometry.r,
                                                           geometry.uKnot, geometry.vKnot, geometry.wKnot,
                                                           geometry.weights.T)

            dRdxi = np.mat(dRdxi)
            dRdeta = np.mat(dRdeta)
            dRdzeta = np.mat(dRdzeta)

            # 计算物理坐标上的基函数微分
            pts = np.mat(geometry.controlPts[sctr])
            # Jacobian matrix
            jacob = pts.T * np.c_[dRdxi.T, dRdeta.T, dRdzeta.T]
            J1 = np.linalg.det(jacob)
            dRdx = np.c_[(dRdxi.T, dRdeta.T, dRdzeta.T)] * jacob.I

            # B matrix
            B = strainDispMatrix3d(nn, dRdx)

            # python 的矩阵赋值和matlab赋值存在一定差异
            # K(sctrB,sctrB) = K(sctrB,sctrB) + B' * C * B * J1 * J2 * wt;

            K_e = B.T * geometry.C * B * J1 * J2 * wt[0]

            for i, sctrB_ind in enumerate(sctrB):
                K[sctrB_ind, sctrB] += K_e[i]
    return K


# 施加边界条件
def f_apply_boundary(f, K, uFixed, vFixed, wFixed, udofs, vdofs, wdofs):
    # a measure of the average size of an element in K used to keep the conditioning of the K matrix
    bcwt = np.mean(np.diag(K))

    # 修改载荷矩阵
    f = f - K[:, udofs] * uFixed.T
    f = f - K[:, vdofs] * vFixed.T
    f = f - K[:, wdofs] * wFixed.T

    f[udofs] = bcwt * uFixed.T
    f[vdofs] = bcwt * vFixed.T
    f[wdofs] = bcwt * wFixed.T

    K[udofs, :] = 0  
    K[vdofs, :] = 0
    K[wdofs, :] = 0
    K[:, udofs] = 0
    K[:, vdofs] = 0
    K[:, wdofs] = 0

    K_udofs = np.eye(len(udofs))
    K_vdofs = np.eye(len(vdofs))
    K_wdofs = np.eye(len(wdofs))

    for i, dofs in enumerate(udofs):
        K[i, udofs] = bcwt * K_udofs[i]
    for i, dofs in enumerate(vdofs):
        K[i, vdofs] = bcwt * K_vdofs[i]
    for i, dofs in enumerate(wdofs):
        K[i, wdofs] = bcwt * K_wdofs[i]

    return f, K
