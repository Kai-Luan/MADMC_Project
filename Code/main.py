import numpy as np
import matplotlib.pyplot as plt
from utils import *
from read_file import *
from indicators import *


numInstance=0
n=200
p=6

w=np.zeros(n,dtype=int) # cout des objets
v=np.zeros((n,p),dtype=int) # utilité v1, v2
filename = "../data/"+"2KP"+str(n)+"-TA-"+str(numInstance)+".dat"

# W: budget
W=readFile(filename,w,v)

#Lecture des point non-dominées (pas present dans le projet)

# filename = "Data/"+str(n)+"_items/2KP"+str(n)+"-TA-"+str(numInstance)+".eff"
# YN=readPoints(filename,p)

# plt.grid()
# plt.scatter(YN[:,0],YN[:,1])

YND=[]  #YND est la liste des solutions non-dominées (approximation)

#Génération de m solutions aléatoires : 
m=100



YND=[] 
A=[]



def PLS(m):
	population = init(m)
	for sol in population:
		plt.scatter(sol[1][0],sol[1][1],color='red')
	Xe = population
	initialisation = Xe.copy()
	Pa = []
	while population:
		voisins = []
		for p in population:
			for candidat in voisinage(p):
				b = False
				for v1, v2 in zip(p[1], candidat[1]):
					if v1 <= v2:
						pass
					else:
						b = True
						break
				if b and miseAJour(Xe, candidat):
					miseAJour(Pa, candidat)
		population = Pa
		Pa = []
	return Xe

def PLS2(m):
	population = init(m)
	population = sorted(population, key= lambda c: (c[1][0], c[1][1]))
	# print("before: ",[y[1] for y in population])
	Pa = []
	Xe = population
	while population:
		for p in population:
			for candidat in voisinage(p):
				b = False
				for v1, v2 in zip(p[1], candidat[1]):
					if v1 <= v2:
						pass
					else:
						b = True
						break
				if b and miseAJour2(Xe, candidat):
					miseAJour2(Pa, candidat)
		population = Pa
		Pa = []
	return Xe

# tabt = []
# propt = []
# dmt=[]

# startt = time.time()
# YND = PLS(m)
# endt = time.time()

# # Append in tab the elements
# tabt.append(endt-startt)
# propt.append(proportion(YN,YND))
# dmt.append(DM(YN,YND,p))

# startt = time.time()
# YND2 = PLS2(m)
# endt = time.time()

# # Append in tab the elements
# tabt.append(endt-startt)
# propt.append(proportion(YN,YND2))
# dmt.append(DM(YN,YND2,p))

# for sol in YND:
# 	plt.scatter(sol[1][0],sol[1][1],color='green')


# plt.show()

#Calcule de la proportion

#print("Proportion = ",proportion(YN,YND))

#Calcule de la distance DM

#print("DM =",DM(YN,YND,p))


# for i in range(len(tabt)):
# 	print("Pour PLS",i+1)
# 	print(f'temps : {tabt[i]}, PYn : {propt[i]}, Dm : {dmt[i]}')

# with open('plsResults.csv', 'w', newline='') as csvfile:
# 	writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
# 	writer.writerow(['','Temps',"PYn","Dm"])
# 	for i in range(len(tabt)):
# 		writer.writerow([f'PLS{i+1}',tabt[i],propt[i],dmt[i]])



















