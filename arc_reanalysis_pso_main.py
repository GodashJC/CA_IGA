'''
对一个简易圆拱模型的重分析优化
JasonChan,HNU
2021,May
'''

from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)  # 小数精度

from getdata import arcdata
from iga_func import *
from gauss_i import gauss_point
from pso2d import PSO
from ca_reanalysis import R_CA
from post_process import *
import time

t1 = time.time()
# 圆拱原始数据
R = 1  # 外径
L = 1  # 高度
t = 0.2  # 厚度
refineCount = 4  # 网格细化次数,h-refinement
# 获取模型信息
geometry = arcdata(R, L, t, refineCount)
original_pts = geometry.controlPts

# # 离散化
generateIGA3DMesh(geometry)
tobemodefied_pts_node = [i for i in range(len(geometry.controlPts)) if
                         -10 <= geometry.controlPts[i][0] <= -9]  # 找到将要优化的点，140个

# PROCESSING
noGPs = 3  # of Gauss points along one direction
[Q, W] = gauss_point(noGPs, 3)
K0 = assembly_K(geometry, Q, W)  # 通过IGA建立初始刚度矩阵
# 施加外力以及边界条件
f = np.mat(np.zeros((geometry.noDofs, 1)))  # 初始化外载荷向量
xConsNodes = [i for i in range(geometry.noCtrPts) if geometry.controlPts[i][0] == 0]
uFixed = np.mat(np.zeros(np.size(xConsNodes)))
vFixed = np.mat(np.zeros(np.size(xConsNodes)))
wFixed = np.mat(np.zeros(np.size(xConsNodes)))
udofs = xConsNodes  # global indecies  of the fixed x disps
vdofs = [node + geometry.noCtrPts for node in xConsNodes]
wdofs = [node + 2 * geometry.noCtrPts for node in xConsNodes]
# external force
# forcedNode = [i for i in range(geometry.noCtrPts) if geometry.controlPts[i][0] == -R
#    and 7 > geometry.controlPts[i][2] > 4]
forcedNode = [i for i in range(geometry.noCtrPts) if geometry.controlPts[i][0] == -R]
f[forcedNode] = -700  # 设置外力
[f, K0] = f_apply_boundary(f, K0, uFixed, vFixed, wFixed, udofs, vdofs, wdofs)

K0 = csr_matrix(K0)
# invK0 = K0.I  # 求初始刚度矩阵的逆矩阵 报错，奇异矩阵 用稀疏矩阵搞
# invK0 = np.linalg.inv(K0.todense()) 报错
# invK0 = linalg.bicg(K0, np.eye(geometry.noDofs)) 报错
invK0 = linalg.inv(K0) #报错
# RuntimeError: failed to factorize matrix at line 110 in file scipy\sparse\linalg\dsolve\SuperLU\SRC\dsnode_bmod.c
'''
彻底没辙   5.13  只能用matlab弄。 python对于奇异矩阵太不友好。
'''

t2 = time.time()
print(t2 - t1)

pop = 25  # 粒子数
generation = 100  # 迭代次数
x_min = [1.05, 1.05]
x_max = [1.15, 1.15]


def fit_fun(x):
    print("迭代成功")
    geometry.controlPts = original_pts
    geometry.controlPts[tobemodefied_pts_node][0] *= x[0]
    geometry.controlPts[tobemodefied_pts_node][1] *= x[1]
    K = assembly_K(geometry, Q, W)
    [ff, K] = f_apply_boundary(f, K, uFixed, vFixed, wFixed, udofs, vdofs, wdofs)
    K = csr_matrix(K)
    disp = R_CA(K, ff, K0, invK0)
    max_stress = post_processing_GetMaxStress(geometry, disp)
    return max_stress


pso = PSO(pop, generation, x_min, x_max, fit_fun)
fit_list, best_pos, gbestFit, gbestFit_list = pso.done()  #

t = [i + 1 for i in range(generation)]
plt.plot(t, gbestFit_list, color='b', linewidth=3)
plt.show()

# SOLVE SYSTEM
# U = linalg.bicg(K0, f)[0]
# # post-processing
# vtsFile = './results/arc.vts'  # 把数值结果的导出到paraview生成云图
# post_processing(geometry, U, vtsFile)


t3 = time.time()
print(t3 - t1)
