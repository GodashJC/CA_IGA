from nurbs_meshing import *
from liner_elastic import strainDispMatrix3d


def post_processing(geometry, U, vtsFile):
    '''
    对每个单元创建8个结点，计算结点上的应力，和位移，用于后续可视化
    '''

    Ux = U[0:geometry.noCtrPts]
    Uy = U[geometry.noCtrPts:2 * geometry.noCtrPts]
    Uz = U[2 * geometry.noCtrPts:geometry.noDofs]

    uKnotVec = np.unique(geometry.uKnot)
    vKnotVec = np.unique(geometry.vKnot)
    wKnotVec = np.unique(geometry.wKnot)

    noKnotsU = len(uKnotVec)
    noKnotsV = len(vKnotVec)
    noKnotsW = len(wKnotVec)

    projcoord = np.c_[
        geometry.controlPts[:, 0] * geometry.weights, geometry.controlPts[:, 1] * geometry.weights, geometry.controlPts[
                                                                                                    :,
                                                                                                    2] * geometry.weights, geometry.weights]

    dim = np.shape(projcoord)[1]

    node = np.zeros((noKnotsU * noKnotsV * noKnotsW, 3))
    count = 0

    for wk in range(noKnotsW):
        zeta = wKnotVec[wk]
        for vk in range(noKnotsV):
            eta = vKnotVec[vk]
            for uk in range(noKnotsU):
                xi = uKnotVec[uk]
                tem = SolidPoint(geometry.noPtsX - 1, geometry.p, geometry.uKnot, geometry.noPtsY - 1, geometry.q,
                                 geometry.vKnot, geometry.noPtsZ - 1, geometry.r, geometry.wKnot, projcoord, dim, xi,
                                 eta, zeta)
                node[count, 0] = tem[0, 0] / tem[0, 3]
                node[count, 1] = tem[0, 1] / tem[0, 3]
                node[count, 2] = tem[0, 2] / tem[0, 3]
                count = count + 1

    # build H8 elements
    chan = np.zeros((noKnotsU, noKnotsW, noKnotsV), int)
    count = 0

    for i in range(noKnotsW):
        for j in range(noKnotsV):
            for k in range(noKnotsU):
                chan[k, i, j] = count
                count = count + 1

    connecU = np.zeros((noKnotsU - 1, 2), int)
    connecV = np.zeros((noKnotsV - 1, 2), int)
    connecW = np.zeros((noKnotsW - 1, 2), int)

    for i in range(np.shape(connecU)[0]):
        connecU[i, :] = [i, i + 1]

    for i in range(np.shape(connecV)[0]):
        connecV[i, :] = [i, i + 1]

    for i in range(np.shape(connecW)[0]):
        connecW[i, :] = [i, i + 1]

    noElems = (noKnotsU - 1) * (noKnotsV - 1) * (noKnotsW - 1)
    elementV = np.zeros((noElems, 8), int)

    e = 0
    for w in range(noKnotsW - 1):
        wConn = connecW[w, :]
        for v in range(noKnotsV - 1):
            vConn = connecV[v, :]
            for u in range(noKnotsU - 1):
                c = 0
                uConn = connecU[u, :]
                for i in range(len(wConn)):
                    for j in range(len(vConn)):
                        for k in range(len(uConn)):
                            elementV[e, c] = chan[uConn[k], wConn[i], vConn[j]]
                            c = c + 1
                e = e + 1

    # renumbering nodes according to Jack Chessa's code
    col3 = elementV[:, 2].copy()
    col4 = elementV[:, 3].copy()
    col7 = elementV[:, 6].copy()
    col8 = elementV[:, 7].copy()

    elementV[:, 2] = col4
    elementV[:, 3] = col3
    elementV[:, 6] = col8
    elementV[:, 7] = col7

    stress = np.zeros((6, noElems, np.shape(elementV)[1]))
    disp = np.zeros((3, noElems, np.shape(elementV)[1]))

    for e in range(noElems):
        idu = geometry.index[e, 0]
        idv = geometry.index[e, 1]
        idw = geometry.index[e, 2]

        xiE = geometry.elRangeU[idu]  # [xi_i,xi_i+1]
        etaE = geometry.elRangeV[idv]
        zetaE = geometry.elRangeW[idw, :]

        sctr = geometry.element[e]  # element scatter vector
        sctrB = np.hstack(
            (sctr, sctr + geometry.noCtrPts, sctr + 2 * geometry.noCtrPts))  # vector that scatters a B matrix
        nn = len(sctr)

        B = np.zeros([6, 2 * nn])
        pts = geometry.controlPts[sctr]

        uspan = findspan(geometry.noPtsX - 1, geometry.p, xiE[0], geometry.uKnot)
        vspan = findspan(geometry.noPtsY - 1, geometry.q, etaE[0], geometry.vKnot)
        wspan = findspan(geometry.noPtsZ - 1, geometry.r, zetaE[0], geometry.wKnot)

        elemDisp = np.c_[Ux[sctr], Uy[sctr], Uz[sctr]]

        # loop over Gauss points
        gp = 0
        for iw in range(0, 2):
            Zeta = zetaE[iw]
            for iv in range(0, 2):
                Eta = etaE[iv]
                for iu in range(0, 2):
                    Xi = xiE[iu]
                    [N, dRdxi, dRdeta, dRdzeta] = NURBS3DBasisDersSpecial(Xi, Eta, Zeta, geometry.p, geometry.q,
                                                                          geometry.r, geometry.uKnot, geometry.vKnot,
                                                                          geometry.wKnot,
                                                                          geometry.weights, uspan, vspan, wspan)

                    # compute the jacobian of physical and parameter domain mapping
                    # then the derivative w.r.t spatial physical coordinates
                    dRdxi = np.mat(dRdxi)
                    dRdeta = np.mat(dRdeta)
                    dRdzeta = np.mat(dRdzeta)

                    jacob = pts.T * np.c_[dRdxi.T, dRdeta.T, dRdzeta.T]

                    #             Jacobian inverse and spatial derivatives
                    dRdx = np.c_[dRdxi.T, dRdeta.T, dRdzeta.T] * jacob.I

                    # B matrix
                    B = strainDispMatrix3d(nn, dRdx)

                    strain = B * (np.mat(U[sctrB]).T)
                    stress[:, e, gp] = list(geometry.C * strain)
                    disp[:, e, gp] = np.mat(N) * elemDisp

                    gp = gp + 1

        col3 = disp[:, e, 2].copy()
        col4 = disp[:, e, 3].copy()
        col7 = disp[:, e, 6].copy()
        col8 = disp[:, e, 7].copy()
        disp[:, e, 2] = col4
        disp[:, e, 3] = col3
        disp[:, e, 6] = col8
        disp[:, e, 7] = col7

    # export to VTK format to plot in Mayavi or Paraview

    numNode = np.shape(node)[0]

    # normal stresses
    sigmaXX = np.zeros((numNode, 1))
    sigmaYY = np.zeros((numNode, 1))
    sigmaZZ = np.zeros((numNode, 1))

    # shear stresses
    sigmaXY = np.zeros((numNode, 1))
    sigmaYZ = np.zeros((numNode, 1))
    sigmaZX = np.zeros((numNode, 1))

    # displacements
    dispX = np.zeros((numNode, 1))
    dispY = np.zeros((numNode, 1))
    dispZ = np.zeros((numNode, 1))

    for e in range(np.shape(elementV)[0]):
        connect = elementV[e, :]
        for i in range(8):
            nid = connect[i]
            sigmaXX[nid] = stress[0, e, i]
            sigmaYY[nid] = stress[1, e, i]
            sigmaZZ[nid] = stress[2, e, i]
            sigmaXY[nid] = stress[3, e, i]
            sigmaYZ[nid] = stress[4, e, i]
            sigmaZX[nid] = stress[5, e, i]

            dispX[nid] = disp[0, e, i]
            dispY[nid] = disp[1, e, i]
            dispZ[nid] = disp[2, e, i]

    # write to VTS 
    # 2021-04-28 显示变形后的模型，需要显示原始结构注掉以下两行
    disp_re = np.c_[dispX, dispY, dispZ]
    node = node + disp_re

    noPtsX = len(np.unique(geometry.uKnot)) - 1
    noPtsY = len(np.unique(geometry.vKnot)) - 1
    noPtsZ = len(np.unique(geometry.wKnot)) - 1
    numNodes = len(node)

    fo = open(vtsFile, "w+")

    fo.write(f'<?xml version="1.0"?>\n<VTKFile type="StructuredGrid" version="0.1" byte_order="BigEndian" >\n\
    <StructuredGrid  WholeExtent="0 {noPtsX} 0 {noPtsY} 0 {noPtsZ}">\n<Piece Extent="0 {noPtsX} 0 {noPtsY} 0 {noPtsZ}">\n\
    <PointData Vectors="Disp"  >\n<DataArray type="Float32" Name="Stress" NumberOfComponents="6" format="ascii">\n')

    for i in range(numNodes):
        fo.write(f'{sigmaXX[i, 0]} {sigmaYY[i, 0]} {sigmaZZ[i, 0]} {sigmaXY[i, 0]} {sigmaYZ[i, 0]} {sigmaZX[i, 0]}\n')

    fo.write('      </DataArray>\n')
    fo.write('      <DataArray type="Float32" Name="Displacement" NumberOfComponents="3" format="ascii">\n')

    for i in range(len(dispX)):
        fo.write(f'   {dispX[i, 0]} {dispY[i, 0]} {dispZ[i, 0]}\n')

    fo.write('      </DataArray>\n</PointData>\n<Celldata>\n</Celldata>\n<Points>\n\
    <DataArray type="Float32" Name="Array" NumberOfComponents="3" format="ascii">\n')

    for i in range(numNodes):
        fo.write(f'  {node[i, 0]} {node[i, 1]} {node[i, 2]}\n')

    fo.write('      </DataArray>\n</Points>\n</Piece> \n</StructuredGrid> \n</VTKFile>\n')

    fo.close()


def post_processing_GetMaxStress(geometry, U):
    '''
    返回最大应力，用于寻优函数
    '''

    Ux = U[0:geometry.noCtrPts]
    Uy = U[geometry.noCtrPts:2 * geometry.noCtrPts]
    Uz = U[2 * geometry.noCtrPts:geometry.noDofs]

    uKnotVec = np.unique(geometry.uKnot)
    vKnotVec = np.unique(geometry.vKnot)
    wKnotVec = np.unique(geometry.wKnot)

    noKnotsU = len(uKnotVec)
    noKnotsV = len(vKnotVec)
    noKnotsW = len(wKnotVec)

    noElems = (noKnotsU - 1) * (noKnotsV - 1) * (noKnotsW - 1)

    stress = np.zeros((6, noElems, 8))
    disp = np.zeros((3, noElems, 8))

    for e in range(noElems):
        idu = geometry.index[e, 0]
        idv = geometry.index[e, 1]
        idw = geometry.index[e, 2]

        xiE = geometry.elRangeU[idu]  # [xi_i,xi_i+1]
        etaE = geometry.elRangeV[idv]
        zetaE = geometry.elRangeW[idw, :]

        sctr = geometry.element[e]  # element scatter vector
        sctrB = np.hstack(
            (sctr, sctr + geometry.noCtrPts, sctr + 2 * geometry.noCtrPts))  # vector that scatters a B matrix
        nn = len(sctr)

        pts = geometry.controlPts[sctr]

        uspan = findspan(geometry.noPtsX - 1, geometry.p, xiE[0], geometry.uKnot)
        vspan = findspan(geometry.noPtsY - 1, geometry.q, etaE[0], geometry.vKnot)
        wspan = findspan(geometry.noPtsZ - 1, geometry.r, zetaE[0], geometry.wKnot)

        elemDisp = np.c_[Ux[sctr], Uy[sctr], Uz[sctr]]

        # loop over Gauss points
        gp = 0
        for iw in range(0, 2):
            Zeta = zetaE[iw]
            for iv in range(0, 2):
                Eta = etaE[iv]
                for iu in range(0, 2):
                    Xi = xiE[iu]
                    [N, dRdxi, dRdeta, dRdzeta] = NURBS3DBasisDersSpecial(Xi, Eta, Zeta, geometry.p, geometry.q,
                                                                          geometry.r, geometry.uKnot, geometry.vKnot,
                                                                          geometry.wKnot,
                                                                          geometry.weights, uspan, vspan, wspan)

                    # compute the jacobian of physical and parameter domain mapping
                    # then the derivative w.r.t spatial physical coordinates
                    dRdxi = np.mat(dRdxi)
                    dRdeta = np.mat(dRdeta)
                    dRdzeta = np.mat(dRdzeta)

                    jacob = pts.T * np.c_[dRdxi.T, dRdeta.T, dRdzeta.T]

                    #             Jacobian inverse and spatial derivatives
                    dRdx = np.c_[dRdxi.T, dRdeta.T, dRdzeta.T] * jacob.I

                    # B matrix
                    B = strainDispMatrix3d(nn, dRdx)

                    strain = B * (np.mat(U[sctrB]).T)
                    stress[:, e, gp] = list(geometry.C * strain)
                    disp[:, e, gp] = np.mat(N) * elemDisp

                    gp = gp + 1

    return max(stress[0, :, :])
