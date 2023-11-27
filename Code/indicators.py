import math
import numpy as np

def proportion(YN,YApprox):
	cpt=0
	for y in YN:	
		for sol in YApprox:
			if np.array_equal(y,sol[1][:]):
				cpt=cpt+1
				break
	return cpt/YN.shape[0]
			

def distanceEuclidienne(y1,y2,poids,p):
	d=0
	for j in range(p):
		d = d + math.sqrt(poids[j] * (y1[j] - y2[j])**2)
	return d 
    
def dprime(YApprox,y,poids,p):    
	minV = 9999999	
	for sol in YApprox:
		dist=distanceEuclidienne(y,sol[1][:],poids,p)
		if dist < minV:
			minV=dist
	return minV
	
	
def DM(YN,YApprox,p):
	Nadir = np.zeros(p,dtype=int)
	Ideal = np.zeros(p,dtype=int)
	for j in range(p):
		Nadir[j]=min(YN[:,j])
	for j in range(p):
		Ideal[j]=max(YN[:,j])
		
	poids = np.zeros(p)
	for j in range(p):
		poids[j] = 1/abs(Ideal[j]-Nadir[j])
	
	d=0
	for y in YN:
		d=d+dprime(YApprox,y,poids,p)
	
	return d/YN.shape[0]
	
