'''
b-spine/nurbs func NURBS相关函数
JasonChan,HNU
2021,March.
reference: The Nurbs Book 2
'''

import numpy as np

# binary search 
def findspan(n,p,u,U):
    '''
    n = m-p-1
    m: {U_0,U_1,......,U_m}
    U: Uknotlist
    '''
    if(u>=U[n+1]):
        return n
    if(u<=U[p]):
        return p
    
    low = p
    high = n+1
    mid = (low+high)//2
    
    while(u<U[mid] or u>=U[mid+1]):
        if( u<U[mid] ):
            high=mid
        else:
            low=mid
        mid=(low+high)//2;
    return mid



def basisfuns(i,u,p,U):
    #we can compute the non zero basis functions at point u, there are p+1 non zero basis functions
    N = [0 for i in range(p+1)]
    N[0] = 1
    left = [0 for i in range(p+1)]
    right = [0 for i in range(p+1)]
    
    for j in range(1,p+1):
        left[j] = u-U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0
        for r in range(j):
            temp = N[r]/(right[r+1]+left[j-r])
            N[r] = saved + right[r+1] * temp
            saved = left[j-r]*temp
        N[j] = saved
    return N



def dersbasisfuns(i,u,p,order,U):
    
    left = [0 for i in range(p+1)]
    right = [0 for i in range(p+1)]
    ndu = [[0 for j in range(p+1)] for i in range(p+1)]
    a = [[0 for j in range(p+1)] for i in range(p+1)]
    ders = [[0 for j in range(p+1)] for i in range(order+1)]
    
    ndu[0][0] = 1
    
    for j in range(1,p+1):
        left[j] = u-U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0
        for r in range(j):
            ndu[j][r] = right[r+1]+left[j-r]
            temp = ndu[r][j-1]/ndu[j][r]
            
            ndu[r][j]=saved+right[r+1]*temp
            saved = left[j-r]*temp
        ndu[j][j]=saved
    
    for j in range(0,p+1):
        ders[0][j] = ndu[j][p];
        
    if(order == 0):
        return ders
    
    for r in range(p+1):
        s1 = 0
        s2 = 1
        a[0][0] = 1
        
        for k in range(1,order+1):
            d = 0;
            rk=r-k
            pk=p-k
            if(r>=k):
                a[s2][0]=a[s1][0]/ndu[pk+1][rk]
                d=a[s2][0]*ndu[rk][pk]
            j1 = 1 if rk >= -1 else -rk 
            j2 = k-1 if (r-1<=pk) else p-r
            
            for j in range(j1,j2+1):
                a[s2][j]=(a[s1][j]-a[s1][j-1])/ndu[pk+1][rk+j]
                d+=a[s2][j]*ndu[rk+j][pk]
            if(r<=pk):
                a[s2][k]= -a[s1][k-1]/ndu[pk+1][r]
                d+=a[s2][k]*ndu[r][pk]
            ders[k][r]=d
            j=s1
            s1=s2
            s2=j 
            
    r = p
    for k in range(1,order+1):
        for j in range(p+1):
            ders[k][j]*=r
        r*=(p-k)

    return ders


def NURBS2Dders(Xi,Eta,p,q,uKnot,vKnot,weight):
    nU = len(uKnot)-1-p-1
    nV = len(vKnot)-1-q-1
    spanU = findspan(nU,p,Xi,uKnot)
    spanV = findspan(nV,q,Eta,vKnot)
    
    N = basisfuns(spanU,Xi,p,uKnot)
    M = basisfuns(spanV,Eta,q,vKnot)
    
    dersN = dersbasisfuns(spanU,Xi,p,nU,uKnot)
    dersM = dersbasisfuns(spanV,Eta,q,nV,vKnot)
    
    uind = spanU - p
    w = 0
    dwdxi = 0
    dwdet = 0
    
    for j in range(0,q+1):
        vind = spanV - q + j
        for i in range(0,p+1):
            c   = uind + i + vind * (nU+1)          
            wgt = weight[c]
            w     += N[i]        * M[j] * wgt
            dwdxi += dersN[1][i] * M[j] * wgt
            dwdet += dersM[1][j] * N[i] * wgt
    #   ders [k] [j] is the kth derivative of the function N_i-p+j,p, 
    #where 0 <= k <= n and 0 <= j <= p.
    
    uind = spanU - p
    k = 0
    dRdxi = [0 for i in range((p+1)*(q+1))]
    dRdet = [0 for i in range((p+1)*(q+1))]
    
    for j in range(0,q+1):
        vind = spanV - q + j
        for i in range(0,p+1):
            c        = uind + i + vind*(nU+1)
            fac      = weight[c]/(w*w)
            dRdxi[k] = (dersN[1][i]*M[j]*w - N[i]*M[j]*dwdxi) * fac
            dRdet[k] = (dersM[1][j]*N[i]*w - N[i]*M[j]*dwdet) * fac
            k += 1;
    
    return dRdxi,dRdet


def NURBS1DBasisDers(xi,p,KnotU,weight):
    n = len(KnotU)-1-p-1
    #N = [0 for i in range(P+1)]
    #dersN = [[0 for j in range(p+1)] for i in range(n+1)]
    spanU = findspan(n,p,xi,KnotU)
    N = basisfuns(spanU,xi,p,KnotU)
    dersN = dersbasisfuns(spanU,xi,p,n,KnotU)
    
    uind  = spanU-p
    w = 0
    dwdxi = 0
    
    for i in range(p+1):
        wgt = weight[uind+i]       
        w += N[i] * wgt
        dwdxi += dersN[1][i]*wgt
    
    R = [0 for i in range(p+1)]
    dRdxi = [0 for i in range(p+1)]
    
    for i in range(p+1):
        fac = weight[uind+i]/(w*w)
        R[i] = N[i]*fac*w
        dRdxi[i] = (dersN[1][i]*w - N[i]*dwdxi) * fac
        
    return R,dRdxi


def NURBS2DBasisDers(Xi,Eta,p,q,uKnot,vKnot,weight):
    nU = len(uKnot)-1-p-1
    nV = len(vKnot)-1-q-1
    spanU = findspan(nU,p,Xi,uKnot)
    spanV = findspan(nV,q,Eta,vKnot)
    
    N = basisfuns(spanU,Xi,p,uKnot)
    M = basisfuns(spanV,Eta,q,vKnot)
    
    dersN = dersbasisfuns(spanU,Xi,p,nU,uKnot)
    dersM = dersbasisfuns(spanV,Eta,q,nV,vKnot)
    
    uind = spanU - p
    w = 0
    dwdxi = 0
    dwdet = 0
    
    for j in range(0,q+1):
        vind = spanV - q + j
        for i in range(0,p+1):
            c   = uind + i + vind * (nU+1)          
            wgt = weight[c]
            w     += N[i]        * M[j] * wgt
            dwdxi += dersN[1][i] * M[j] * wgt
            dwdet += dersM[1][j] * N[i] * wgt
    #   ders [k] [j] is the kth derivative of the function N_i-p+j,p, 
    #where 0 <= k <= n and 0 <= j <= p.
    
    uind = spanU - p
    k = 0
    R = [0 for i in range((p+1)*(q+1))]
    dRdxi = [0 for i in range((p+1)*(q+1))]
    dRdet = [0 for i in range((p+1)*(q+1))]
    
    for j in range(0,q+1):
        vind = spanV - q + j
        for i in range(0,p+1):
            c        = uind + i + vind*(nU+1)
            fac      = weight[c]/(w*w)
            R[k]     = N[i]*M[j]*fac*w
            dRdxi[k] = (dersN[1][i]*M[j]*w - N[i]*M[j]*dwdxi) * fac
            dRdet[k] = (dersM[1][j]*N[i]*w - N[i]*M[j]*dwdet) * fac
            k += 1
    
    return R,dRdxi,dRdet

def NURBS2DBasisDersSpecial(Xi,Eta,p,q,uKnot,vKnot,weight,spanU,spanV):
    nU = len(uKnot)-1-p-1
    nV = len(vKnot)-1-q-1
    N = basisfuns(spanU,Xi,p,uKnot)
    M = basisfuns(spanV,Eta,q,vKnot) 
    dersN = dersbasisfuns(spanU,Xi,p,nU,uKnot)
    dersM = dersbasisfuns(spanV,Eta,q,nV,vKnot)
    
    uind = spanU - p
    w = 0
    dwdxi = 0
    dwdet = 0

    for j in range(0,q+1):
        vind = spanV - q + j
        for i in range(0,p+1):
            c   = uind + i + vind * (nU+1)          
            wgt = weight[c]
            w     += N[i]        * M[j] * wgt
            dwdxi += dersN[1][i] * M[j] * wgt
            dwdet += dersM[1][j] * N[i] * wgt    
            
    uind = spanU - p
    k = 0
    R = [0 for i in range((p+1)*(q+1))]
    dRdxi = [0 for i in range((p+1)*(q+1))]
    dRdet = [0 for i in range((p+1)*(q+1))]       
            
    for j in range(0,q+1):
        vind = spanV - q + j
        for i in range(0,p+1):
            c        = uind + i + vind*(nU+1)
            fac      = weight[c]/(w*w)
            R[k]     = N[i]*M[j]*fac*w
            dRdxi[k] = (dersN[1][i]*M[j]*w - N[i]*M[j]*dwdxi) * fac
            dRdet[k] = (dersM[1][j]*N[i]*w - N[i]*M[j]*dwdet) * fac
            k += 1
    
    return R,dRdxi,dRdet


def NURBS3DBasisDers(Xi,Eta,Zeta,p,q,r,uKnot,vKnot,wKnot,weight):
    nU = len(uKnot)-1-p-1
    nV = len(vKnot)-1-q-1
    nW = len(wKnot)-1-r-1
    spanU = findspan(nU,p,Xi,uKnot)
    spanV = findspan(nV,q,Eta,vKnot)
    spanW = findspan(nW,r,Zeta,wKnot)
    
    N = basisfuns(spanU,Xi,p,uKnot)
    M = basisfuns(spanV,Eta,q,vKnot)
    P = basisfuns(spanW,Zeta,r,wKnot)
    
    dersN = dersbasisfuns(spanU,Xi,p,nU,uKnot)
    dersM = dersbasisfuns(spanV,Eta,q,nV,vKnot)
    dersP = dersbasisfuns(spanW,Zeta,r,nW,wKnot)
    
    uind = spanU - p
    w = 0
    dwdxi = 0
    dwdet = 0
    dwdze = 0
    
    for k in range(0,r+1):
        wind = spanW - r + k
        for j in range(0,q+1):
            vind = spanV - q + j
            for i in range(0,p+1):
                c   = uind + i + (nU+1) * ((nV+1)*wind + vind)
                wgt = weight[c]
                w     += N[i]        * M[j] * P[k] * wgt
                dwdxi += dersN[1][i] * M[j] * P[k] * wgt
                dwdet += dersM[1][j] * N[i] * P[k] * wgt
                dwdze += dersP[1][k] * N[i] * M[j] * wgt
    #   ders [k] [j] is the kth derivative of the function N_i-p+j,p, 
    #where 0 <= k <= n and 0 <= j <= p.
    
    uind = spanU - p
    kk = 0
    R = [0 for i in range((p+1)*(q+1)*(r+1))]
    dRdxi = [0 for i in range((p+1)*(q+1)*(r+1))]
    dRdet = [0 for i in range((p+1)*(q+1)*(r+1))]
    dRdze = [0 for i in range((p+1)*(q+1)*(r+1))]
    
    for k in range(r+1):
        wind = spanW - r + k
        for j in range(q+1):
            vind = spanV - q + j
            for i in range(p+1):
                c        = uind + i + (nU+1) * ( (nV+1)*wind + vind )
                fac      = weight[c]/(w*w)
                nmp      = N[i]*M[j]*P[k] 
                R[kk]     = nmp * fac * w
                dRdxi[kk] = (dersN[1][i]*M[j]*P[k]*w - nmp*dwdxi) * fac
                dRdet[kk] = (dersM[1][j]*N[i]*P[k]*w - nmp*dwdet) * fac
                dRdze[kk] = (dersP[1][k]*N[i]*M[j]*w - nmp*dwdze) * fac   
                kk      += 1
    
    return R,dRdxi,dRdet,dRdze


def NURBS3DBasisDersSpecial(Xi,Eta,Zeta,p,q,r,uKnot,vKnot,wKnot,weight,spanU,spanV,spanW):
    nU = len(uKnot)-1-p-1
    nV = len(vKnot)-1-q-1
    nW = len(wKnot)-1-r-1
    N = basisfuns(spanU,Xi,p,uKnot)
    M = basisfuns(spanV,Eta,q,vKnot)
    P = basisfuns(spanW,Zeta,r,wKnot) 
    dersN = dersbasisfuns(spanU,Xi,p,nU,uKnot)
    dersM = dersbasisfuns(spanV,Eta,q,nV,vKnot)
    dersP = dersbasisfuns(spanW,Zeta,r,nW,wKnot)
    uind = spanU - p
    w = 0
    dwdxi = 0
    dwdet = 0
    dwdze = 0
    
    for k in range(0,r+1):
        wind = spanW - r + k
        for j in range(0,q+1):
            vind = spanV - q + j
            for i in range(0,p+1):
                c   = uind + i + (nU+1) * ( (nV+1)*wind + vind)
                wgt = weight[c]
                w     += N[i]        * M[j] * P[k] * wgt
                dwdxi += dersN[1][i] * M[j] * P[k] * wgt
                dwdet += dersM[1][j] * N[i] * P[k] * wgt
                dwdze += dersP[1][k] * N[i] * M[j] * wgt
            
    uind = spanU - p
    kk = 0
    R = [0 for i in range((p+1)*(q+1)*(r+1))]
    dRdxi = [0 for i in range((p+1)*(q+1)*(r+1))]
    dRdet = [0 for i in range((p+1)*(q+1)*(r+1))]       
    dRdze = [0 for i in range((p+1)*(q+1)*(r+1))]  
    
    for k in range(0,r+1):
        wind = spanW - r + k
        for j in range(0,q+1):
            vind = spanV - q + j
            for i in range(0,p+1):
                c        = uind + i + (nU+1) * ( (nV+1)*wind + vind)
                fac      = weight[c]/(w*w)
                nmp      = N[i]*M[j]*P[k]
                
                R[kk]     = nmp * fac * w;
                dRdxi[kk] = (dersN[1][i]*M[j]*P[k]*w - nmp*dwdxi) * fac
                dRdet[kk] = (dersM[1][j]*N[i]*P[k]*w - nmp*dwdet) * fac
                dRdze[kk] = (dersP[1][k]*N[i]*M[j]*w - nmp*dwdze) * fac
                
                kk      += 1
    
    return R,dRdxi,dRdet,dRdze


