#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirac Leptogenesis in Left-Right Symmetric Models
Author: Ajay Kaladharan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 20:29:27 2025

@author: ajayk

Notation consistent with arXiv:1503.00629 and Leptogenesis and low-energy CP violation in
a type-II-dominated left-right seesaw model
"""

import numpy as np
import math as mt
import pandas as pd
from scipy.special import kn #Modified bessels equation K_n(x)=kn(n,x)
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class LRSM1E3():
    def __init__(self,Y,f,ME1,ME2,ME3,mphiR,mphiL,kR,mh1,mh2,thetap):
        self.mphiR=mphiR
        self.mphiL=mphiL
        self.ME1=ME1
        self.ME2=ME2
        self.ME3=ME3
        self.ME=np.array([[self.ME1,0,0],[0,self.ME2,0],[0,0,self.ME3]])
        self.IME=np.array([[1/self.ME1,0,0],[0,1/self.ME2,0],[0,0,1/self.ME3]])
        self.kR=kR
        self.mh1=mh1 # charged scalar mass at zero temeprature. Used to fit neutrino mass
        self.mh2=mh2
        self.thetap=thetap # mixing angle for charged scalars
        self.Yuk=0.4
        "Nueutrino oscillation data"
        self.eV=10**(-9) #eV to GeV conversion
        self.s12sq=np.array([0.307,0.012,0.011]) # Central value, error
        self.s23sq=np.array([0.572,0.018,0.023])
        self.s13sq=np.array([0.02203,0.00056,0.00058])
        self.deltaCP_n=np.array([197,41,25])
        self.Dm21sq_n=np.array([7.41*10**(-5),0.21*10**(-5),0.20*10**(-5)])
        self.Dm31sq_n=np.array([2.511*10**(-3),0.027*10**(-3),0.027*10**(-3)])
        "Asymmetry"
        self.Y_obs=np.array([8.579*10**(-11), 0.109*10**(-11),0.109*10**(-11)])
        # Yukawa matrices
        self.Y=Y
        self.YdY=self.MatM(self.Dagger(self.Y),self.Y) #Y^\dagger Y
        self.f=f
        self.fdf=self.MatM(self.Dagger(self.f),self.f) #f^\dagger f
        # Defining constants
        self.gsm  = 106.75 #SM dof
        self.Mpl  = 1.22*10**(19) #Planck Mass
        CGF = np.double(1.1663787*10**(-5))
        self.vL=np.double(mt.sqrt(1/(mt.sqrt(2)*CGF)))
        self.kL=self.vL/np.sqrt(2)
        # The masses of SM fermions
        self.me=np.array([0.5109989461*10**(-3),0.00031*10**(-3),0.0000000031*10**(-3)]) #In GeV
        self.mmu=np.array([105.6583745*10**(-3),0.00024*10**(-3),0.0000024*10**(-3)])  #In GeV
        self.mtau=np.array([1776.86*10**(-3),0.12*10**(-3),0.12*10**(-3)])     #In GeV
        # Computing masses of SM fermions. For crosscheck
        self.ml,self.Ul=self.Mlepton()  # Ul unitarity matrix that diagonalise charged lepton mass
        # Finding Mass of heavy lepton after right hand symmetry breaking
        self.MEp,self.VLT,self.VRT=self.MassofE()
        self.M1=self.MEp[0]
        self.M2=self.MEp[1]
        self.M3=self.MEp[2]
        self.H1=self.H(self.M1)
        # Calculating Yukawa matrices in mass basis
        self.fL=self.MatM(self.f,self.VLT)
        self.fR=self.MatM(self.f,self.VRT)
        self.YL=self.MatM(self.Y,self.VLT)
        self.fLdfL=self.MatM(self.Dagger(self.fL),self.fL)
        self.fRdfR=self.MatM(self.Dagger(self.fR),self.fR)
        self.YldYl=self.MatM(self.Dagger(self.YL),self.YL)
        # Calculating the decay width
        fac=self.M1/(16*mt.pi)
        G1=fac*(self.YldYl[0][0]+self.fLdfL[0][0]+self.fRdfR[0][0])
        self.G1=G1.real
        
        fac=self.M2/(16*mt.pi)
        G2=fac*(self.YldYl[1][1]+self.fLdfL[1][1]+self.fRdfR[1][1])
        self.G2=G2.real
        
        fac=self.M3/(16*mt.pi)
        G3=fac*(self.YldYl[2][2]+self.fLdfL[2][2]+self.fRdfR[2][2])
        self.G3=G3.real
        
        # Calculating CP assymmetry
        self.eps1=self.epsilon(0)
        self.eps2=self.epsilon(1)
        self.eps3=self.epsilon(2)
        #self.eps1=10**(-6)
        # Calculating Branching ratio
        self.Br1nuR, self.Br1eR, self.Br1phiL, self.Br1chiL=self.findBR(0)
        self.Br2nuR, self.Br2eR, self.Br2phiL, self.Br2chiL=self.findBR(1)
        self.Br3nuR, self.Br3eR, self.Br3phiL, self.Br3chiL=self.findBR(2)
        self.Yeqfac=15/(8*np.pi**2*self.gsm) #Need to add g_i and 2 for bosons. For m=0
        # Setting Neutrino mass
        self.mE,self.VL,self.VR=self.MassofE0() # Mass of heavy leptons at zero temperatures
        # Initialising for neutrino fit
        self.t13=0
        self.t12=0
        self.t23=0
        self.deltamu=0
        self.deltatau=0
        self.deltaCP=0
        self.deltae=0
        self.varphi1=0
        self.varphi2=0
        self.Mnu=self.Neuutrinomassmastrix()
        self.mnusq,self.UPMNS=self.PMNSmatrix()
        self.PMNSextract(self.UPMNS)
        # Lambda_5
        self.L5=self.Lambda5()
        self.GammaphiR_phiL,self.GammaphiR_phiL_H=self.GammaphiRphiL()  # the rate and ratio with H
        self.GammaphiR_eR,self.GammaphiR_eR_ratio=self.GammaphiReR()
        self.Wratio=self.GammaphiR_phiL/self.GammaphiR_eR   # The washout ratio, requires to be more that 10
        
        
        
        
        
        
        
    def Dagger(self,X):
        "Dagger of X (3*3) matrix"
        M=np.zeros([3,3], dtype=complex)
        for i in range(3):
            for j in range(3):
                M[i][j]=np.conj(X[j][i])
        return M 
    
    def Transpose(self,X):
        "Transpose of X 3*3 matrix"
        M=np.zeros([3,3], dtype=complex)
        for i in range(3):
            for j in range(3):
                M[i][j]=X[j][i]
        return M
        
    
    def MatM(self,X,Y):
        "Matrix multiplication return M=XY, X and Y are 3*3 matrix"
        M=np.zeros([3,3], dtype=complex)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    M[i][j]+=X[i][k]*Y[k][j]
        return M
    
    def MatM3(self,X,Y,Z):
        "Matrix multiplication return M=XYZ, X,Y and Z are 3*3 matrix"
        M=np.zeros([3,3], dtype=complex)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        M[i][j]+=X[i][k]*Y[k][l]*Z[l][j]
        return M

    def Mat_full(self):
        M=np.zeros([6,6], dtype=complex)
        for i in range(3):
            for j in range(3,6):
                M[i][j]=self.Y[i][j-3]*self.kR
        Ydagger=self.Dagger(self.Y)
        for i in range(3,6):
            for j in range(3):
                M[i][j]=Ydagger[i-3][j]*self.kL
        for i in range(3,6):
            for j in range(3,6):
                M[i][j]=self.ME[i-3][j-3]
        return M
    
    def my_kn1(self,x):
        """
        Convenience wrapper for kn(1, x)
        """
        return kn(1, x) if x<=600 else 1e-100#3. + 8.*x

    def my_kn2(self,x):
        """
        Convenience wrapper for kn(2, x)
        """
        return kn(2, x) if x<=600 else 1e-100#15. + 8.*x
    
    def H(self,T):
        "Hubble expansion rate H(T)"
        y=1.66*mt.sqrt(self.gsm)*(T*T/self.Mpl)
        return y
    
    def rhoR(self):
        "Check \rho^2 is small"
        rho=self.MatM(self.Y,self.IME)*self.kR
        #print(rho)
        el=rho.flatten
        return rho
    
    def rhoL(self):
        "Check \rho^2 is small"
        rho=self.MatM(self.Y,self.IME)*self.kL
        #print(rho)
        el=rho.flatten
        return el
    
    def Mlepton(self):
        "Mass of leptons"
        IMAT=self.IME*self.kL*self.kR
        ydagger=self.Dagger(self.Y)
        ML=self.MatM3(self.Y,IMAT,ydagger)
        ml, mye=np.linalg.eig(ML)
        "Note: Dagger(mye).ML.mye is diagonal. So returning mye, such that mye. M_{diag}. mye^\dagger =ML"
        sor=np.argsort(ml.real)
        myeT=self.Transpose(mye)
        ml1=ml[sor]
        myeT1=myeT[sor]
        mye1=self.Transpose(myeT1)
        return ml1.real, mye1
    
    def MassofE(self):
        "Mass of heavy leptons"
        Mat1=(self.kR**2 * self.MatM3(self.Dagger(self.Y), self.Y, self.IME)).astype(np.complex128)
        Mat=(self.ME+0.5*Mat1).astype(np.complex128)
        mE, mye=np.linalg.eig(Mat)
        mE1=abs(mE.real)
        sor=np.argsort(mE1)
        mE2=mE1[sor]
        MMd=(self.MatM(Mat,self.Dagger(Mat))).astype(np.complex128)
        mEN1,mye1=np.linalg.eig(MMd)
        myeT=self.Transpose(mye1)
        sor=np.argsort(abs(mEN1.real))
        mEN1=mEN1[sor]
        myeT1=myeT[sor]
        VL=self.Transpose(myeT1)
        MMd=(self.MatM(self.Dagger(Mat),Mat)).astype(np.complex128)
        mEN1,mye1=np.linalg.eig(MMd)
        sor=np.argsort(abs(mEN1.real))
        myeT=self.Transpose(mye1)
        mEN1=mEN1[sor]
        myeT1=myeT[sor]
        VR=self.Transpose(myeT1)
        VL1=self.Dagger(VL)
        VR1=self.Dagger(VR) 
        return mE2,VL1,VR1
    
    def MassofE0(self):
        "Mass of heavy leptons at zero temperature"
        "Should return mE, VL and VR"
        Mat1=(self.kR**2*self.MatM3(self.Dagger(self.Y),self.Y,self.IME)).astype(np.complex128)
        Mat2=(self.kL**2*(self.MatM3(self.Y,self.Dagger(self.Y),self.IME))).astype(np.complex128)
        Mat=(self.ME+0.5*(Mat1+Mat2)).astype(np.complex128)
        mE, mye=np.linalg.eig(Mat)
        MMd=(self.MatM(Mat,self.Dagger(Mat))).astype(np.complex128)
        mE1,mye1=np.linalg.eig(MMd)
        VL=self.Dagger(mye1)
        MdM=(self.MatM(self.Dagger(Mat),Mat)).astype(np.complex128)
        mE2,mye2=np.linalg.eig(MdM)
        VR=self.Dagger(mye2)
        return mE.real, VL, VR
    
    def epsilon(self,i):
        "Calculating CP asymmetry. i is the index" 
        e=0
        MI=self.MEp[i]
        fac=1/(8*mt.pi)
        fac1=1/(self.fLdfL[i][i]+self.fRdfR[i][i]+self.YldYl[i][i])
       
        for k in range(3):
            if (i!=k):
                MK=self.MEp[k]
                t1=MI**2/(MI**2-MK**2)
                term1=t1*(np.imag(self.fRdfR[k][i]*self.YldYl[i][k]))
                e+=term1
        y=fac*fac1*e
        return y.real
   
    def findBR(self,i):
        "Branching ratio for E  \rightarrow \bar{\nu}_R \phi_R^- "
        num=self.fRdfR[i][i]
        den=2*self.fLdfL[i][i]+2*self.fRdfR[i][i]+2*self.YldYl[i][i]
        BrnuR=(num/den).real
        num2=self.fLdfL[i][i]
        BreR=(num/den).real
        BrphiL=(2*num2/den).real
        num1=2*self.YldYl[i][i]
        BrchiL=(num1/den).real
        return BrnuR, BreR, BrphiL, BrchiL
    
    def entropy(self,T):
        "s=g_\star 2 \pi^2 T^3/45"
        y=self.gsm*2*np.pi**2*T*T*T/45
        return y
    
    def Yieq(self,z):
        "Y_i^{eq}. Need to add g_i factor"
        #There is not double in Blotzmann equation. So g_i=1 for all case
        y=45*z*z*self.my_kn2(z)/(4*np.pi**4*self.gsm)
        return y
    
    def Gamma1(self,z):
        "reaction rate of decay of E_1 without entropy factor"
        #g_{E}=1. #check 1903.12192
        Y1eq=self.Yieq(z)
        Y=self.G1*Y1eq*self.my_kn1(z)/self.my_kn2(z)
        return Y
    
    def Gamma2(self,z):
        "reaction rate of decay of E_2 without entropy factor"
        fac=self.M2/self.M1
        Y1eq=self.Yieq(fac*z)
        Y=self.G2*Y1eq*self.my_kn1(fac*z)/self.my_kn2(fac*z)
        return Y
    
    def Gamma3(self,z):
        "reaction rate of decay of E_3 without entropy factor"
        fac=self.M3/self.M1
        Y1eq=self.Yieq(fac*z)
        Y=self.G3*Y1eq*self.my_kn1(fac*z)/self.my_kn2(fac*z)
        return Y
    
    def GammaphiRphiL(self):
        "return magnitude and ratio with Hubble at mphi_R"
        HT=self.H(self.mphiR)
        fac=(1-self.mphiL**2/self.mphiR**2)
        num=self.L5**2*self.kR**2
        den=16*np.pi**2*self.mphiR
        GphiR=fac*num/den
        ratio=GphiR/HT
        return GphiR, ratio
    
    def GammaphiReR(self):
        " \phi_R^- -> \e_R +\nu_R.  Return magnitude and ratio with Hubble at mphi_R of the largest magnitude"
        rho_R=self.MatM(self.Y,self.IME)*self.kR
        U=self.MatM(self.f,self.Dagger(rho_R))
        UdU=self.MatM(U,self.Dagger(U))
        TrUdU=UdU[0][0]+UdU[1][1]+UdU[2][2]
        GphiR=TrUdU.real*self.mphiR/(8*np.pi)
        HT=self.H(self.mphiR)
        ratio=GphiR/HT
        return GphiR, ratio
    
    def scaterr_EE_phiphi(self,s,ME):
        "E \bar{E} to phi_R phi_R"
        fac=self.Yuk**4/(96*np.pi)
        t1=np.sqrt(1-(4*ME**2/s))
        t2=1+(2*ME**2/s)
        y=fac*t1*t2
        return y
    
    def scaterr_EE_ff(self,s,ME):
        "E \bar{E} to f f"
        fac=119*self.Yuk**4/(288*np.pi)
        t1=np.sqrt(1-(4*ME**2/s))
        t2=1+(2*ME**2/s)
        y=fac*t1*t2
        return y
    
    def scaterr_EE_gamgam(self,s,ME):
        "E \bar{E} to gam gam"
        fac=self.Yuk**4/(4*np.pi)
        term=np.sqrt(1-(4*ME**2/s))
        t1=(1+(4*ME**2/s)-(16*ME**4/s**2))*np.log((1+term)/(1-term))
        t2=2*(1+(4*ME**2/s))*term
        y=fac*(t1+t2)
        return y
    
    def gamma_EE(self,z,ME):
        """
        \frac{\gamma(z)}{sH(z)z}=\frac{45 z^3 M}{g_\star 128 \pi^6}\int_{x_{min}}^{\infty}dx\sqrt{x}\mathcal{K}_1(z\sqrt{x})\hat{\sigma}(x)
        x=s/M^2
        """
        fac=(ME**4)/(64*np.pi**4*z)
        def integrand(x):
            s=x*ME**2
            scat_gamgam=self.scaterr_EE_gamgam(s,ME)
            scat_EE_ff=self.scaterr_EE_ff(s,ME)
            scaterr_EE_phiphi=self.scaterr_EE_phiphi(s,ME)
            sigma=scat_gamgam+scat_EE_ff+scaterr_EE_phiphi
            k1=self.my_kn1(z*np.sqrt(x))
            y=np.sqrt(x)*k1*sigma
            return y
        
        integral, err=quad(integrand,4,np.inf)
        gamma=fac*integral
        return gamma
        
    
    
        
        
    
    def Bolteqn(self,y,z):
        """
        Notation
        y[0]=\Sigma E_1
        y[1]=\Delta E_1
        y[2]=\Sigma E_2
        y[3]=\Delta E_2
        y[4]=\Sigma E_3
        y[5]=\Delta E_3
        y[6]=\Sigma \nu_R
        y[7]=\Delta \nu_R
        y[8]=\Delta \phi_R^-
        y[9]=\Delta e_R
        y[10]=\Delta \phi_R^--
        y[11]=\Delta \psi_L
        y[12]=\Delta \phi_L
        """
        SE1=y[0]
        DE1=y[1]
        SE2=y[2]
        DE2=y[3]
        SE3=y[4]
        DE3=y[5]
        SnuR=y[6]
        DnuR=y[7]
        DphiR=y[8]
        Der=y[9]
        DphiR2m=y[10]
        DpsiL=y[11]
        DphiL=y[12]
        
        T=self.M1/z # Temperature
        H=self.H(T) #Hubble rate
        "Equilibirum number density values"
        YE1eq=self.Yieq(z)
        fac2=self.M2/self.M1
        YE2eq=self.Yieq(fac2*z)
        fac3=self.M3/self.M1
        YE3eq=self.Yieq(fac3*z)
        YphiReq=self.Yieq((self.mphiR/self.M1)*z)
        Ynureq=self.Yeqfac
        Yereq=self.Yeqfac
        YpsiLeq=2*self.Yeqfac
        YphiLeq=2*self.Yieq((self.mphiL/self.M1)*z)
        # Boltzmann equation
        M=np.zeros(13)
        # Decay rate
        g1=self.Gamma1(z)
        g2=self.Gamma2(z)
        g3=self.Gamma3(z)
        "Equation for \Sigma E_1"
        M[0]=-g1*((SE1/YE1eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br1nuR))*(z/self.H1)
        fac_sHz=(z/self.H1)*(45*z*z*z/(self.gsm*2*np.pi**2*self.M1**3))   # This factorial compared to 1/(sHz)
        scat_gamma=self.gamma_EE(z,self.M1)
        M[0]+=-fac_sHz*scat_gamma*((SE1**2/(4*YE1eq**2))-1)
        "Equation for \Delta E_1"
        term=-self.eps1*g1*((SnuR/Ynureq)-2)
        term+=g1*self.Br1nuR*(-(DE1/YE1eq)-(DnuR/Ynureq)+(SnuR/(2*Ynureq))*(DphiR/YphiReq))
        term+=g1*self.Br1eR*(-(DE1/YE1eq)-(Der/Yereq)+(DphiR2m/YphiReq))
        term+=g1*self.Br1chiL*(-(DE1/YE1eq)+(DpsiL/YpsiLeq))
        term+=g1*self.Br1phiL*(-(DE1/YE1eq)-(DpsiL/YpsiLeq)+(DphiL/YphiLeq))
        M[1]=term*(z/self.H1)
        
        "Equation for \Sigma E_2"
        M[2]=-g2*((SE2/YE2eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br2nuR))*(z/self.H1)
        fac_sHz=(z/self.H1)*(45*z*z*z/(self.gsm*2*np.pi**2*self.M1**3))   # This factorial compared to 1/(sHz)
        scat_gamma=self.gamma_EE(z*fac2,self.M2)
        M[2]+=-fac_sHz*scat_gamma*((SE2**2/(4*YE2eq**2))-1)
        "Equation for \Delta E_2"
        term=-self.eps2*g2*((SnuR/Ynureq)-2)
        term+=g2*self.Br2nuR*(-(DE2/YE2eq)-(DnuR/Ynureq)+(SnuR/(2*Ynureq))*(DphiR/YphiReq))
        term+=g2*self.Br2eR*(-(DE2/YE2eq)-(Der/Yereq)+(DphiR2m/YphiReq))
        term+=g2*self.Br2chiL*(-(DE2/YE2eq)+(DpsiL/YpsiLeq))
        term+=g2*self.Br2phiL*(-(DE2/YE2eq)-(DpsiL/YpsiLeq)+(DphiL/YphiLeq))
        M[3]=term*(z/self.H1)
        
        "Equation for \Sigma E_3"
        M[4]=-g3*((SE3/YE3eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br3nuR))*(z/self.H1)
        fac_sHz=(z/self.H1)*(45*z*z*z/(self.gsm*2*np.pi**2*self.M1**3))   # This factorial compared to 1/(sHz)
        scat_gamma=self.gamma_EE(z*fac3,self.M3)
        M[4]+=-fac_sHz*scat_gamma*((SE3**2/(4*YE3eq**2))-1)
        "Equation for \Delta E_3"
        term=-self.eps3*g3*((SnuR/Ynureq)-2)
        term+=g3*self.Br3nuR*(-(DE3/YE3eq)-(DnuR/Ynureq)+(SnuR/(2*Ynureq))*(DphiR/YphiReq))
        term+=g3*self.Br3eR*(-(DE3/YE3eq)-(Der/Yereq)+(DphiR2m/YphiReq))
        term+=g3*self.Br3chiL*(-(DE3/YE3eq)+(DpsiL/YpsiLeq))
        term+=g3*self.Br3phiL*(-(DE3/YE3eq)-(DpsiL/YpsiLeq)+(DphiL/YphiLeq))
        M[5]=term*(z/self.H1)
        
        "Equation for \Sigma nu_R"
        M[6]=g1*self.Br1nuR*((SE1/YE1eq)-(SnuR/Ynureq))*(z/self.H1)
        M[6]+=g2*self.Br2nuR*((SE2/YE2eq)-(SnuR/Ynureq))*(z/self.H1)
        M[6]+=g3*self.Br3nuR*((SE3/YE3eq)-(SnuR/Ynureq))*(z/self.H1)
        
        "Equation for \Delta nu_R"
        term1=-self.eps1*g1*((SE1/YE1eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br1nuR))
        term1+=-self.eps2*g2*((SE2/YE2eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br2nuR))
        term1+=-self.eps3*g3*((SE3/YE3eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br3nuR))
        term2=-g1*self.Br1nuR*((DnuR/Ynureq)-(SnuR/(2*Ynureq))*(DphiR/YphiReq)+(DE1/YE1eq))
        term2+=-g2*self.Br2nuR*((DnuR/Ynureq)-(SnuR/(2*Ynureq))*(DphiR/YphiReq)+(DE2/YE2eq))
        term2+=-g3*self.Br3nuR*((DnuR/Ynureq)-(SnuR/(2*Ynureq))*(DphiR/YphiReq)+(DE3/YE3eq))
        M[7]=(term1+term2)*(z/self.H1)
        
        
        "Equation for \Delta \phi^-_R"
        #Its just negative of M[3]
        term1=self.eps1*g1*((SE1/YE1eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br1nuR))
        term1+=self.eps2*g2*((SE2/YE2eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br2nuR))
        term1+=self.eps3*g3*((SE3/YE3eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br3nuR))
        term2=g1*self.Br1nuR*((DnuR/Ynureq)-(SnuR/(2*Ynureq))*(DphiR/YphiReq)+(DE1/YE1eq))
        term2+=g2*self.Br2nuR*((DnuR/Ynureq)-(SnuR/(2*Ynureq))*(DphiR/YphiReq)+(DE2/YE2eq))
        term2+=g3*self.Br3nuR*((DnuR/Ynureq)-(SnuR/(2*Ynureq))*(DphiR/YphiReq)+(DE3/YE3eq))
        M[8]=(term1+term2)*(z/self.H1)
        
        
        "Equation for \Delta e_R"
        term1=-self.eps1*g1*((SE1/YE1eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br1nuR))
        term1+=-self.eps2*g2*((SE2/YE2eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br2nuR))
        term1+=-self.eps3*g3*((SE3/YE3eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br3nuR))
        term2=-g1*self.Br1eR*((Der/Yereq)-(DphiR2m/YphiReq)+(DE1/YE1eq))
        term2+=-g2*self.Br2eR*((Der/Yereq)-(DphiR2m/YphiReq)+(DE2/YE2eq))
        term2+=-g3*self.Br3eR*((Der/Yereq)-(DphiR2m/YphiReq)+(DE3/YE3eq))
        M[9]=(term1+term2)*(z/self.H1)
        
        
        "Equation for \Delta \phi_R^{--}"
        # Just negative of M4
        term1=self.eps1*g1*((SE1/YE1eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br1nuR))
        term1+=self.eps2*g2*((SE2/YE2eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br2nuR))
        term1+=self.eps3*g3*((SE3/YE3eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br3nuR))
        term2=g1*self.Br1eR*((Der/Yereq)-(DphiR2m/YphiReq)+(DE1/YE1eq))
        term2+=g2*self.Br2eR*((Der/Yereq)-(DphiR2m/YphiReq)+(DE2/YE2eq))
        term2+=g3*self.Br3eR*((Der/Yereq)-(DphiR2m/YphiReq)+(DE3/YE3eq))
        M[10]=(term1+term2)*(z/self.H1)
        
        
        "Equation for \Delta \psi_L"
        epspsi1=-2*self.eps1
        term1=epspsi1*g1*((SE1/YE1eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br1nuR))
        epspsi2=-2*self.eps2
        term1+=epspsi2*g2*((SE2/YE2eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br2nuR))
        epspsi3=-2*self.eps3
        term1+=epspsi3*g3*((SE3/YE3eq)-2*(1+((SnuR/(2*Ynureq))-1)*self.Br3nuR))
        term2=-g1*self.Br1chiL*(-(DE1/YE1eq)+(DpsiL/YpsiLeq))
        term2+=-g2*self.Br2chiL*(-(DE2/YE2eq)+(DpsiL/YpsiLeq))
        term2+=-g3*self.Br3chiL*(-(DE3/YE3eq)+(DpsiL/YpsiLeq))
        term3=-g1*self.Br1phiL*((DE1/YE1eq)+(DpsiL/YpsiLeq)-(DphiL/YphiLeq))
        term3+=-g2*self.Br2phiL*((DE2/YE2eq)+(DpsiL/YpsiLeq)-(DphiL/YphiLeq))
        term3+=-g3*self.Br3phiL*((DE3/YE3eq)+(DpsiL/YpsiLeq)-(DphiL/YphiLeq))
        M[11]=(term1+term2+term3)*(z/self.H1)
        
        
        "Equation for \phi_L"
        term1=g1*self.Br1phiL*((DE1/YE1eq)+(DpsiL/YpsiLeq)-(DphiL/YphiLeq))
        term1+=g2*self.Br2phiL*((DE2/YE2eq)+(DpsiL/YpsiLeq)-(DphiL/YphiLeq))
        term1+=g3*self.Br3phiL*((DE3/YE3eq)+(DpsiL/YpsiLeq)-(DphiL/YphiLeq))
        M[12]=term1*(z/self.H1)
        #print("z=",z," y=",y)
        #print("M=",M)
        return M
    
    def Lambda5(self):
        "Evaluate lambda_5 from charge scalar mass and mixing angle"
        tan2t=np.tan(2*self.thetap)
        massdiff=self.mphiR**2-self.mphiL**2
        L5=tan2t*massdiff/(2*self.kL*self.kR)
        return L5

    

   
    
    def Bolteqn1(self,z,y):
        return self.Bolteqn(y,z)
    
    def solving1(self):
        iT=self.kR
        zinitial=self.M1/iT
        fac2=self.M2/self.M1
        fac3=self.M3/self.M1
        zfinal=10**(2)
        xfinal=2*zinitial
        """
        return values for z, N_{N_i}(z), N_{\alpha \beta}(z) 
        Using solve_ivp function
        """
        "Initial value"
        X0=np.zeros(13) # equilibrium value
        X0[0]=2*self.Yieq(zinitial)
        X0[2]=2*self.Yieq(fac2*zinitial)
        X0[4]=2*self.Yieq(fac3*zinitial)
        X0[6]=2*self.Yeqfac
        sol = solve_ivp(self.Bolteqn1,[zinitial,zfinal], X0, method='BDF')
        z=sol.t
        y=sol.y
        return z, y

 
    
    def assymetry(self):
        fac=28/79
        z,y=self.solving1()
        SE1=y[0]
        DE1=y[1]
        SE2=y[2]
        DE2=y[3]
        SE3=y[4]
        DE3=y[5]
        SnuR=y[6]
        DnuR=y[7]
        DphiR=y[8]
        Der=y[9]
        DphiR2m=y[10]
        DpsiL=y[11]
        DphiL=y[12]
        Len=len(DnuR)
        assym=fac*DnuR[Len-1]
        return assym
        
    
    def Neuutrinomassmastrix(self):
        "Return neutrino mass matrix"
        gev=10**9 #coverting to eV
        M=np.zeros([3,3], dtype=complex)
        fvL=self.MatM(self.f,self.VL)
        fvR=self.MatM(self.f,self.VR)
        stp=np.sin(self.thetap)
        ctp=np.cos(self.thetap)
        for a in range(3):
            for b in range(3):
                for i in range(3):
                    fac1=self.mE[i]*stp*ctp/(16*np.pi**2)
                    fac2=(self.mh2**2/(self.mh2**2-self.mE[i]**2))*np.log(self.mh2**2/self.mE[i]**2)
                    fac2+=-(self.mh1**2/(self.mh1**2-self.mE[i]**2))*np.log(self.mh1**2/self.mE[i]**2)
                    M[a][b]+=fvL[a][i]*np.conj(fvR[b][i])*fac1*fac2*gev
        U=self.Dagger(self.Ul)
        Mt=self.MatM3(U,M,self.Dagger(U)) #In the basis charged leptons are diagonal
        return Mt  
    
    def PMNSmatrix(self):
        "Obtain PMNS matrix"
        Mnu=self.Mnu
        Mnusq=self.MatM(Mnu,self.Dagger(Mnu))
        mnusq, mye=np.linalg.eig(Mnusq)
        "Note: Dagger(mye).Mnu.mye is diagonal. So returning mye as PMNS matrix. Check Eq:4.8 2205.09127"
        mnusq1,mye1=self.normalordering(mnusq.real, mye)
        return mnusq1, mye1

    def GammaphiLnuLeL(self):
        "Three body decay of \phi_L^-"
        fac=(5/192)*(1/128*np.pi**3)
        sum=0
        for i in range (3):
            sumf=0
            sumy=0
            for alpha in range(3):
                sumf+=np.abs(self.f[alpha][i])**2
                sumy+=np.abs(self.Y[alpha][i])**2
            sum+=sumf*sumy*(self.mh2**3/self.MEp[i]**2)
        y=fac*sum
        return y
    
    def normalordering(self,mnusq,mye):
        "Rotate to get normal ordering m1<m2<m3"
        sor=np.argsort(mnusq)
        myeT=self.Transpose(mye)
        mnusq1=mnusq[sor]
        myeT1=myeT[sor]
        mye1=self.Transpose(myeT1)
        return mnusq1,mye1
    
    def invertedordering(self,mnusq,mye):
        "Rotate to get inverted ordering m3<m1<m2"
        sor=np.argsort(mnusq)
        sor1=np.array([sor[1],sor[2],sor[0]])
        mnusq1=mnusq[sor1]
        mye1=mye[sor1]
        return mnusq1,mye1
        
            
    
    def PMNSextract(self,U):
        "Extract the elements for the PMNS matrix"
        self.t13=np.arcsin(abs(U[0][2]))
        c13=np.cos(self.t13)
        s13=np.sin(self.t13)
        if (abs(U[0][0])==0):
            self.t12=np.pi/2
        else:
            self.t12=np.arctan(abs(U[0][1])/abs(U[0][0]))
        if (abs(U[2][2])==0):
            self.t23=np.pi/2
        else:
            self.t23=np.arctan(abs(U[1][2])/abs(U[2][2]))
        c23=np.cos(self.t23)
        s23=np.sin(self.t23)
        c12=np.cos(self.t12)
        s12=np.sin(self.t12)
        self.deltamu=np.angle(U[1][2])
        self.deltatau=np.angle(U[2][2])
        term1=(np.conj(U[0][0])*U[0][2]*U[2][0]*np.conj(U[2][2]))/(c12*c13**2*c23*s13)
        term2=c12*c23*s13
        deltaCP=-np.angle((term1+term2)/(s12*s23))
        deltaCP=deltaCP*(180/np.pi)
        if (deltaCP<0):
            deltaCP=360+deltaCP
        self.deltaCP=deltaCP    
        self.deltae=np.angle(np.e**(self.deltaCP*1j)*U[0][2])
        self.varphi1=2*np.angle(np.e**(self.deltae*1j)*np.conj(U[0][0]))
        self.varphi2=2*np.angle(np.e**(self.deltae*1j)*np.conj(U[0][1]))
        return None
    
    def pull(self,val,exp,expm,expp):
        y=(val-exp)/(expm+expp)
        return y**2
    
    def chiwashout1(self):
        "The process \phi_R -> e_R \nu_R should be out of equilibirum"
        print("GammaphiR_eR_ratio",self.GammaphiR_eR_ratio)
        if (self.GammaphiR_eR_ratio<0.1):
            y=0
        else:
            y=self.pull(self.GammaphiR_eR_ratio,0.1,0.01,0.01)
        return y

    def chiwashout2(self):
        "The process \phi_R -> \phi_L \chi_L should be in of equilibirum"
        print("GammaphiR_phiL=",self.GammaphiR_phiL)
        if (self.GammaphiR_phiL>10):
            y=0
        else:
            y=self.pull(self.GammaphiR_phiL,10,1,1)
        return y  
    
    def chiwashout3(self):
        "The ratio between washout process should be greater than 100"
        print("Wratio=",self.Wratio)
        if (self.Wratio>100):
            y=0
        else:
            y=self.pull(self.Wratio,100,10,10)
        return y  
    
    def E1yuk(self):
        "would like ydy[0][0] to be very small"
        ydy=self.YdY[0][0]
        logydy=np.log10(ydy.real)
        if (logydy<-6):
            y=0
        else:
            y=self.pull(logydy,-6,0.5,0.5)
        return y

    def Fyuk(self):
        "To constrain fdf"
        fdf=np.array([self.fdf[0][0].real,self.fdf[1][1].real,self.fdf[2][2].real])
        maxfdf=max(fdf)
        logmax=np.log10(maxfdf)
        if (logmax<-6):
            y=0
        else:
            y=self.pull(logmax,-6,0.5,0.5)
        return y    
        
        
    
    
    def chisq(self):
        s12sq=np.sin(self.t12)**2
        s13sq=np.sin(self.t13)**2
        s23sq=np.sin(self.t23)**2
        assym=self.assymetry()
        chi=0
        chilep=chi
        chi+=self.pull(s12sq,self.s12sq[0],self.s12sq[1],self.s12sq[2])
        chi+=self.pull(s13sq,self.s13sq[0],self.s13sq[1],self.s13sq[2])
        chi+=self.pull(s23sq,self.s23sq[0],self.s23sq[1],self.s23sq[2])
        dm21sq=self.mnusq[1]-self.mnusq[0]
        dm31sq=self.mnusq[2]-self.mnusq[0]
        chi+=self.pull(dm21sq,self.Dm21sq_n[0],self.Dm21sq_n[1],self.Dm21sq_n[2])
        chi+=self.pull(dm31sq,self.Dm31sq_n[0],self.Dm31sq_n[1],self.Dm31sq_n[2])
        chi+=self.pull(self.deltaCP,self.deltaCP_n[0],self.deltaCP_n[1],self.deltaCP_n[2])
        chi+=self.pull(assym,self.Y_obs[0],self.Y_obs[1],self.Y_obs[2])
        return np.log10(chi)
    
    


"""
Benchmark point 1

Y=np.array([[-1.21115719e-03+2.09070528e-08j,  1.49641781e-06-2.39318335e-05j,
         4.22749825e-08-1.48752380e-05j],
       [ 8.17557437e-06-1.56569494e-05j, -5.24568688e-03-7.99687832e-03j,
         8.66432362e-03+1.56076082e-02j],
       [ 4.26066125e-05-1.21050462e-03j, -6.84273100e-02-2.34196183e-03j,
        -4.08635802e-02-1.58083622e-03j]])

f=np.array([[ 0.00032585-1.85003140e-04j,  0.00029032+1.29433592e-04j,
        -0.00029298-5.90613431e-04j],
       [-0.00028874+1.88184541e-03j,  0.00060055-5.34830890e-05j,
        -0.00037153+2.31640565e-05j],
       [-0.00021716-1.58874298e-03j,  0.0002084 -7.06157581e-06j,
        -0.00043318+7.75233950e-04j]])

thetap=3.546774691916593e-10

kR=10**13
x1=0.5
x2=0.6
x3=0.7
ME1=x1*kR
ME2=x2*kR
ME3=x3*kR
mphiR=10**9
mphiL=10**7
mh1=10**7
mh2=10**9
"""

"""
Benchmark point 2

f=np.array([[ 3.622400548167377e-04-1.005003086446003e-04j,
   2.415879522161632e-06-3.872737285863342e-04j,
  -5.543430082325818e-04+5.196154710230735e-05j],
 [-2.687261262915792e-04-1.070343491257327e-03j,
  -5.828392525479732e-05+1.441756336110794e-03j,
   2.868958761037886e-04-1.481470359101655e-03j],
 [-2.688136259205309e-04-4.136876852586184e-04j,
  -2.046285568082913e-04-1.580355077630064e-04j,
   1.000946190030654e-03+1.464229838357423e-05j]])

Y=np.array([[9.38075332e-04+1.58043614e-07j, 1.11007118e-05+2.86620092e-05j,
        9.03591714e-06+2.46819070e-05j],
       [2.67821636e-04+3.74112205e-04j, 3.01423472e-02-1.33776797e-02j,
        1.12066911e-02-2.95155813e-02j],
       [4.80188705e-04+6.62063719e-04j, 4.85695877e-02-3.98769738e-02j,
        3.23744957e-02-1.13494079e-02j]])

thetap=4.2727119226357576e-10

kR=10**13
x1=0.3
x2=0.6
x3=0.9
ME1=x1*kR
ME2=x2*kR
ME3=x3*kR
mphiR=10**9
mphiL=10**7
mh1=10**7
mh2=10**9
"""

"""

"""

