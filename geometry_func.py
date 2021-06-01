'''
建立一个几何体类，包含控制点，节点表，权重等信息
'''

class Geometry:
    def __init__(self, controlPts, uKnot, vKnot, wKnot, weights, p, q, r, noPtsX, noPtsY, noPtsZ, C):
        self.controlPts = controlPts
        self.uKnot = uKnot
        self.vKnot = vKnot
        self.wKnot = wKnot
        self.weights = weights
        self.p = p
        self.q = q
        self.r = r
        self.noPtsX = noPtsX
        self.noPtsY = noPtsY
        self.noPtsZ = noPtsZ
        self.noCtrPts = noPtsX * noPtsY * noPtsZ
        self.noDofs = self.noCtrPts * 3
        self.C = C

# geometry.
