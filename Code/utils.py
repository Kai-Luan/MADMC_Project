import numpy as np
from read_file import *
from indicators import *
from gurobipy import GRB
import gurobipy as gp

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
	"""
	m: le nombre de solution à générer
	n: dimension de la solution
	p: le nombre de critères
	v: utilité des n objets
	w: poids des n objets
	W: la taille du sac
	"""
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
				vStart+=v[arr[i]]
		res.append([xStart, vStart])
	return res

# Initialisation generalisee

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
def voisinage(x, n, v, w, W):
	"""
	x: solution à voisiner
	n: dimension de la solution
	v: utilité des n objets
	w: poids des n objets
	W: la taille du sac
	"""
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

# ======================== ND-Tree ================================
class Node():
	def __init__(self, y, pere=None):
		self.pi = y
		self.pn = y
		self.points = [y]
		self.pere = pere

	def isLeaf(self):
		return not isinstance(self.points[0], Node)
	
	def isEmpty(self):
		return len(self.points) == 0
	
	def closest(self, y):
		y = np.array(y)
		P = [(x.pi+x.pn)/2 for x in self.points]
		P = np.array(P)
		D = np.linalg.norm(P-y, axis=1)
		index = np.argmin(D)
		return self.points[index]
	
	def remove(self, z):
		for i,p in enumerate(self.points):
			if np.all(z==p):
				self.points.pop(i)
				break
	
	def updateIdealNadir(self, y):
		b = 0
		points = []
		if self.isLeaf(): points = self.points
		else:
			for x in self.points:
				points.append(x.pi)
				points.append(x.pn)
		a = np.max(points, 0)
		b = np.min(points, 0)
		if np.any(b < self.pn) or np.any(a > self.pi):
			self.pi = a
			self.pn = b
			if self.pere is not None: self.pere.updateIdealNadir(y)

	def updateNode(self, tree, y):
		"""
		return False if y is dominated, else True
		"""
		if np.all(self.pn >= y): return False
		elif np.all(y >= self.pi): 
			# Remove self and sub_tree of n
			if self.pere is not None: self.pere.remove(self)
			return True
		elif np.all(self.pi >= y) or np.all(y >= self.pn):
			if self.isLeaf(): 
				# on supprime les solutions dominées
				# sinon, le candidat y s'il est dominé
				L = []
				for z in self.points:
					if np.all(z >= y): return False
					elif not np.all(y > z): L.append(z)
				self.points = L
			else: 
				for x in self.points:
					if not x.updateNode(tree, y): return False
					elif x.isEmpty(): self.remove(x)
				# supprime les noeuds à un successeur
				if len(self.points) == 1:
					node = self.points[0]
					if self.pere is not None:
						self.pere.remove(self)
						self.pere.append(node)
					else: tree.root = node
		return True

	def split(self, nChild=2):
		points = np.array(self.points)
		D = [
			np.linalg.norm(points - p, axis=1).mean()
			for p in points
		]
		I = np.argsort(D)[-nChild:]
		I.sort()
		
		N, Z = [], []
		for i in I[::-1]:
			z = self.points.pop(i)
			N.append(Node(z, pere=self))
			Z.append(z)
		Z = np.array(Z)
		P = self.points
		self.points = N
		while P:
			z = P.pop()
			tmp = self.closest(z)
			tmp.points.append(z)
			tmp.updateIdealNadir(z)
	
	def insert(self, y, NBMAX, nChild):
		if self.isLeaf():
			self.points.append(y)
			self.updateIdealNadir(y)
			if len(self.points) > NBMAX:
				self.split(nChild)
		else:
			self.closest(y).insert(y, NBMAX,nChild)

class NDTree():
	def __init__(self, NBMAX=20):
		self.root = None
		self.NBMAX = NBMAX

	# return (YND, Squares)
	# YND non dominated solutions
	# Squares: (depth, ideal point, nadir point) of each node
	def getPoints(self, leafOnly = True):
		def get(node,count=0):
			if node.isLeaf():
				return node.points, [(count,node.pi, node.pn)]
			else:
				L = []
				M = [] if leafOnly else [(count,node.pi, node.pn)]
				for x in node.points: 
					ynd, yid = get(x, count+1)
					L.extend(ynd)
					M.extend(yid)
				return L, M
		return get(self.root)
	
	def update(self,y):
		dim = len(y)
		if self.root is None:
			self.root = Node(y)
			return True
		elif self.root.updateNode(self, y):
			self.root.insert(y, self.NBMAX, nChild=dim)
			return True
		return False

# MR: max regret: x in O
# PMR: pairwise max regret, (x,y) in O
# MMR: minimax regret: x

# CSS:
# x: minimax regret
# y: argmax PMR(x,y)

class Model():
	def __init__(self, dim):
		self.model = None
		self.dim = dim

	# OWA aggregator model
	def init_owa_model(self):
		self.model = gp.Model('ModelEU')
		m = self.model
		w = m.addVars(*range(self.dim))
		self.w = np.array(w)
		for i in range(self.dim-1):
			m.addConstr(w[i]-w[i+1]>=0, f'c{i+1}')
		m.addConstr(sum(w) ==1)

	def update_owa(self, a,b):
		a, b = np.asarray(a.sorted()), np.asarray(b.sorted())
		self.model.addConstr(sum((a-b)*self.w) >= 0)

	# Optimize with the OWA function
	def optimize(self, a, b=None):
		if b is None: b = np.zeros(self.dim)
		a, b = np.asarray(a), np.asarray(b)
		a.sort()
		b.sort()
		self.model.setObjective(sum(self.w*(a-b)), GRB.MAXIMIZE)
		self.model.update()
		self.model.optimize()
		return self.model.ObjVal()
	
	def CSS(self, X):
		MR = [self.optimize(x) for x in X]
		i = np.argmin(MR)
		PMR = [self.optimize(y, X[i]) for y in X]
		j = np.argmax(PMR)
		return (X[i],X[j])

class Aggreg():
	# Aggregation functions
	def owa(self, y, alpha):
		y = np.asarray(y)
		y.sort()
		return (y*alpha).sum()
	
	def eu(self, y, alpha):
		y = np.asarray(y)
		return (y*alpha).sum()
	
	def choquet(self, y, alpha):
		return 0


	


		