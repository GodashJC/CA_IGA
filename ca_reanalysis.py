# CA组合近似法 缩减基法
import numpy as np
from scipy.sparse import linalg


def R_CA(kk, ff, kk0, invkk0):
    sdof = len(ff)
    deltk = kk - kk0
    print(deltk)
    nb = 11
    rb = np.mat(np.zeros((sdof, nb)))
    rb[:, 0] = invkk0 * ff
    for i in range(1, nb):
        vec1 = deltk * rb[:, i - 1]
        vec2 = invkk0 * vec1
        rb[:, i] = -vec2
    kkr = rb.T * kk * rb
    ffr = rb.T * ff
    #     z = ffr*kkr.I  #可能报错点
    z = linalg.bicg(kkr, ffr)[0]
    z = np.mat(z)
    disp = rb * z.T
    return disp
