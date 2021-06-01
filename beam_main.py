
'''
对一个简易梁模型的等几何分析
JasonChan,HNU
2021,May

'''
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import time

from getdata import beam3dC1Data
from iga_func import *
from gauss_i import gauss_point
from post_process import post_processing

t1 = time.time()
# 原始数据
a = 10  # 设置梁的长宽高
b = 4
c = 2
noPtsX = 21  # 设置梁的控制点数
noPtsY = 7
noPtsZ = 5
p = 2
q = 2
r = 2

# 获取模型信息
geometry = beam3dC1Data(a, b, c, noPtsX, noPtsY, noPtsZ, p, q, r)

# 离散化
generateIGA3DMesh(geometry)

# PROCESSING
noGPs = 3  # of Gauss points along one direction
[Q, W] = gauss_point(noGPs, 3)
K = assembly_K(geometry, Q, W)

# initialization
f = np.mat(np.zeros((geometry.noDofs, 1)))  # external force vector

#查找边界结点

leftNodes = [i for i in range(geometry.noCtrPts) if geometry.controlPts[i][0] == 0]
rightNodes = [i for i in range(geometry.noCtrPts) if geometry.controlPts[i][0] == a]  # a=10
# essential boundary conditions
uFixed = np.mat(np.zeros(np.size(leftNodes)))
vFixed = np.mat(np.zeros(np.size(leftNodes)))
wFixed = np.mat(np.zeros(np.size(leftNodes)))

rightNodes_array = np.array(rightNodes) + 2 * geometry.noCtrPts
f[rightNodes_array] = -50

udofs = leftNodes  # global indecies  of the fixed x disps
vdofs = [Nodes + geometry.noCtrPts for Nodes in leftNodes]  # global indecies  of the fixed y disps
wdofs = [Nodes + 2 * geometry.noCtrPts for Nodes in leftNodes]  # global indecies  of the fixed z disps

# Computing external force and apply boundary conditions
[f, K] = f_apply_boundary(f, K, uFixed, vFixed, wFixed, udofs, vdofs, wdofs)

# # SOLVE SYSTEM
K_sparse = csr_matrix(K)  # sparse_matrix
U = linalg.bicg(K_sparse, f)[0]

# post-processing
vtsFile = './results/3Dbeam_res01.vts'  # 把数值结果的导出到paraview生成云图
post_processing(geometry, U, vtsFile)


t2 = time.time()
print(t2-t1)
