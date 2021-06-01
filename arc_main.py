
'''
一简易几何结构的等几何分析
JasonChan,HNU
2021,May.
'''

from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np

np.set_printoptions(precision=4)  # 小数精度

from getdata import arcdata
from iga_func import *
from gauss_i import gauss_point
from post_process import post_processing
import time

t1 = time.time()
# 圆拱原始数据
R = 1  # 外径
L = 1  # 高度
t = 0.2  # 厚度
refineCount = 4  # 网格细化次数,h-refinement
# 获取模型信息
geometry = arcdata(R, L, t, refineCount)

# # 离散化
generateIGA3DMesh(geometry)

# PROCESSING
noGPs = 4  # of Gauss points along one direction
[Q, W] = gauss_point(noGPs, 3)
K = assembly_K(geometry, Q, W)

# 施加外力和边界条件
f = np.mat(np.zeros((geometry.noDofs, 1)))  # 外载荷向量
xConsNodes = [i for i in range(geometry.noCtrPts) if geometry.controlPts[i][0] == 0]
uFixed = np.mat(np.zeros(np.size(xConsNodes)))
vFixed = np.mat(np.zeros(np.size(xConsNodes)))
wFixed = np.mat(np.zeros(np.size(xConsNodes)))

udofs = xConsNodes  # global indecies  of the fixed x disps
vdofs = [node + geometry.noCtrPts for node in xConsNodes]
wdofs = [node + 2 * geometry.noCtrPts for node in xConsNodes]

# external force
#forcedNode = [i for i in range(geometry.noCtrPts) if geometry.controlPts[i][0] == -R
          #    and 7 > geometry.controlPts[i][2] > 4]

forcedNode = [i for i in range(geometry.noCtrPts) if geometry.controlPts[i][0] == -R]
f[forcedNode] = -700 #设置外力

[f, K] = f_apply_boundary(f, K, uFixed, vFixed, wFixed, udofs, vdofs, wdofs)

# SOLVE SYSTEM
K_sparse = csr_matrix(K)  # sparse_matrix
# invK = linalg.inv(K_sparse) 报错 求逆方法何在 ？？
U = linalg.bicg(K_sparse, f)[0]

# post-processing
vtsFile = './results/arc01.vts'  # 把数值结果的导出到paraview生成云图
post_processing(geometry, U, vtsFile)

t2 = time.time()
print(t2 - t1)
