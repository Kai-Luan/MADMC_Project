import numpy as np
from read_file import *
from indicators import *

def miseAJour(X, x):
	if X:
		p = len(X[0])	
		for y in X:
			b = True
			for i in range(p):
				if y[1][i] >= x[1][i]: pass
				else:
					b = False
					break		
			if b: return False
		X.append(x)
		i = 0
		n = len(X)
		while i < n:
			y = X[i]
			b = True
			for j in range(p):
				if x[1][j] > y[1][j]: pass
				else:
					b = False
					break	
			if b: 
				X.pop(i)
				n -= 1
			else: i+=1
		return True	
	else:
		X.append(x)
		return True

def discrete_binary_search(tab, lo, hi, x):
	while lo < hi:
		mid = lo + (hi - lo) // 2
		if x[1][0] <= tab[mid][1][0]:
			hi = mid
		else:
			lo = mid + 1
	return lo

# On suppose X triés croissant, et bi-ojectif
def miseAJour2(X, x):
	if X:
		p = len(X[0])	
		index = discrete_binary_search(X, 0, len(X), x)
		for y in X[index:]:
			# y domine faiblement x
			if y[1][1] >= x[1][1]:
				return False
		# print("INDEX: ", index, len(X), x[1])
		# print("X ",[y[1] for y in X])
		X.insert(index, x)
		# print("X ",[y[1] for y in X])
		i = 0
		while i < index:
			y = X[i]
			if x[1][1] > y[1][1]:
				# print("POP: ", x[1], y[1])
				X.pop(i)
				index -= 1
			else: i+=1
		return True	
	else:
		X.append(x)
		return True


def init(m, n, p, v, w, W):
	res = []
	def rapport(v1, v2, poids, q):
		return (q*v1 +(1-q)*v2)/poids
	
	for _ in range(m):
		q = np.random.random()
		xStart=np.zeros(n,dtype=int) # n solutions
		arr = [rapport(v[j][0], v[j][1], w[j], q) for j in range(n)]
		arr = np.argsort(arr)[::-1]
		wTotal=0
		vStart=np.zeros(p,dtype=int) # objectifs
		for i in range(n):
			if wTotal+w[arr[i]]<=W:
				xStart[arr[i]]=1
				wTotal=wTotal+w[arr[i]]
				for j in range(p):
					vStart[j]=vStart[j]+v[arr[i],j]
		res.append([xStart, vStart])
	return res




# GENERALISATION 

# generalized version, it takes a vector allv and a vector q
def rapport_gen(allv, poids, q):
	# ([v1, ..., vn] * [q1, ..., qn])/poids
	return np.sum((q*allv))/poids

# Initialisation de n solutions
# m : solutions aléatoires, w : vecteur poids, W : taille du sac, v : vecteur valeurs, n : nombre d'objets, p : nombre d'objectifs.
def init_gen(m,w,W,v,n,p):
	res = []
	for _ in range(m):
		#q = np.random.random()
		# generalized q (instead of bi-distrib)
		q = np.random.dirichlet(np.ones(6),size=1)[0]
		xStart=np.zeros(200,dtype=int) # n solutions
		# on range dans l'ordre decroissant
		#arr = [rapport(v[j][0], v[j][1], w[j], q) for j in range(n)]
		arr = [rapport(v[j], w[j], q) for j in range(n)]
		arr = np.argsort(arr)[::-1]

		# total mass of the bag
		wTotal=0

		# initial objective values
		vStart=np.zeros(p,dtype=int) # objectifs

		# we loop on all the items
		for i in range(n):
			# On prend les objets selon le rapport jusqu'à ce qu'il n'y a plus de place
			if wTotal+w[arr[i]]<=W:
				xStart[arr[i]]=1
				wTotal=wTotal+w[arr[i]]
				for j in range(p):
					vStart[j]=vStart[j]+v[arr[i],j]
		res.append([xStart, vStart])
	return res

# Fonction de voisinage
def voisinage(x,n, v, w, W):
	res = []
	poids = sum(w*x[0])
	for i in range(n):
		if x[0][i] == 0: continue
		for j in range(n):
			if x[0][j] == 1: continue
			if poids - w[i] + w[j] > W: continue
			copyx, copyv = x[0].copy(), x[1].copy()
			copyx[i] = 0
			copyx[j] = 1
			copyv = copyv - v[i] + v[j]
			res.append([copyx, copyv])
	return res

class NDTree():
	def __init__(self, pi, pn, points, nmax):
		self.pi = pi
		self.pn = pn
		self.points = points
		self.succ = []
		self.count = len(points)
		self.nmax = nmax


# Maximization
def updateNode(node, y, pere):
	if node.pn >= y: return False
	elif y >= node.pi: 
		# Remove node and sub_tree of n
		return True
	elif node.pi >= y or y >= node.pn:
		for x in node.succ:
			if not updateNode(x, y): 
				return False
			else:
				if x.isEmpty():
					node.remove(x)
		if len(node.succ) == 1:
			pere.remove(node)
			pere.append(node.succ)
	else: # skip this node
		pass
	return True

def updateIdealNadir(node, y):
	pass

def insert(node, y):
	if node.isLeaf():
		node.points.append(y)
		updateIdealNadir(node, y)
	

		
