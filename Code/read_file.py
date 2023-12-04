import numpy as np

# w: weigh of object
# v: values of objects (2 objectives), now generalized
def readFile(filename,w,v):
	f = open(filename, "r")
	i=0
	for line in f:
		if line[0]=="i":
			data = line.split()
			w[i]=int(data[1])
			for index in range(2,len(data)):
				# v[i,0]=int(data[2])
				# v[i,1]=int(data[3])
				# on commence à 0 avec index-2
				v[i,index-2]=int(data[index])
			i=i+1
		else:
			if line[0]=="W":
				data = line.split()	
				W=int(data[1])
	f.close()
	return W

# p: number of objectives
def readPoints(filename, p):
	f = open(filename, "r")
	nbPND = 0
	for line in f: nbPND += 1
	YN = np.zeros((nbPND,p))
	f = open(filename, "r")
	i=0
	for line in f:
		data = line.split()
		for j in range(p):
			YN[i][j]=int(data[j])
		i=i+1
	f.close()
	return YN



