'''
计算应变矩阵
'''
import numpy as np

def strainDispMatrix3d(nn,dRdx):
    B = np.zeros((6,nn*3))
    B[0,0:nn] = dRdx[:,0].T
    B[1,nn:2*nn] = dRdx[:,1].T
    B[2,2*nn:3*nn]  = dRdx[:,2].T

    B[3,0:nn] = dRdx[:,1].T
    B[3,nn:2*nn] = dRdx[:,0].T

    B[4,2*nn:3*nn] = dRdx[:,1].T
    B[4,nn:2*nn] = dRdx[:,2].T

    B[5,0:nn] = dRdx[:,2].T
    B[5,2*nn:3*nn] = dRdx[:,0].T

    return np.mat(B)