import numpy as np
from read_file import *
from indicators import *
from gurobipy import GRB
import gurobipy as gp
from tqdm import tqdm
from time import time
from functools import partial
from multiprocessing import Process, Pool
import multiprocessing.managers
from functools import reduce

#file:///Users/christian/Desktop/MADMC_Project/projetMADMC.pdf

# Initialisation generalisee
# generalized version, it takes a vector allv and a vector q
def rapport(allv, poids, q):
	# ([v1, ..., vn] * [q1, ..., qn])/poids
	return np.sum((q*allv))/poids

# Initialisation de n solutions
# m : solutions aléatoires, w : vecteur poids, W : taille du sac, v : vecteur valeurs, n : nombre d'objets, p : nombre d'objectifs.
def init(m,params):
	(n,p,v,w,W) = params
	res = []
	for _ in range(m):
		# generalized q (instead of bi-distrib)
		q = np.random.dirichlet(np.ones(p),size=1)[0]
		xStart=np.zeros(n,dtype=int) # n solutions
		# on range dans l'ordre decroissant
		arr = [rapport(v[j], w[j], q) for j in range(n)]
		arr = np.argsort(arr)[::-1]

		# total mass of the bag
		wTotal=0
		# initial objective values
		vStart=np.zeros(p,dtype=int) # objectifs

		# we loop on all the items
		for i in arr:
			# On prend les objets selon le rapport jusqu'à ce qu'il n'y a plus de place
			if wTotal+w[i]<=W:
				xStart[i]=1
				wTotal+=w[i]
				vStart+=v[i]
		res.append([xStart, vStart])
	return res

# Fonction de voisinage
def voisinage(x, params):
	"""
	x: solution à voisiner
	params: (n,_,v,w,W)
		n: dimension de la solution,
		v: utilité des n objets,
		w: poids des n objets,
		W: la taille du sac
	"""
	(n,_,v,w,W) = params
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

def PLS(m, params, NBMAX= 20, verbose=False):
	population = init(m,params)
	Xe = NDTree(NBMAX=NBMAX)
	for e in population: Xe.update(e)
	Pa = NDTree(NBMAX=NBMAX)

	iteration = 1
	while population:
		if verbose: 
			print(f'{iteration = } | population size: {len(population)}')
			population = tqdm(population)
		for p in population:
			voisins = voisinage(p, params)
			for candidat in voisins:
				if np.all(p[1] >= candidat[1]): continue
				if Xe.update(candidat): Pa.update(candidat)
		population = Pa.getPoints()
		Pa.reset()
		iteration += 1
	return Xe.getPoints()

# ======================== ND-Tree ================================
class Node():
	# Un noeud du ND-Tree
	"""
 	pi: point ideal
        pn: point nadir
	points: lui meme et fils
        pere: noeud parent
	toremove: les points a ne pas garder
 	"""
	def __init__(self, y, pere=None):
		self.pi = y[1]
		self.pn = y[1]
		self.points = [y]
		self.pere = pere
		self.toremove = []
	# verifie si c'est un ensemble de solution
	def isLeaf(self):
		return not isinstance(self.points[0], Node)
	# verifie si l'ensemble des solutions est vide
	def isEmpty(self):
		return self.points == []
	# quand on ajoute une solution
	def closest(self, y):
		P = np.array([(x.pi+x.pn)/2 for x in self.points])
		D = np.linalg.norm(P-y, axis=1)
		return self.points[np.argmin(D)]
	# solution a retiree car dominee
	def remove(self, z):
		self.toremove.append(z)
	# met a jour le noeud
	def refresh(self):
		if self.toremove:
			self.points = [x for x in self.points if x not in self.toremove]
			self.toremove = []
	# z: point
	# obj: un autre point
	# on remplace z par obj
	def replace(self, z, obj):
		for i,p in enumerate(self.points):
			if z is p: 
				self.points[i] = obj
				break
	# ajoute le point a l'ensemble
	def append(self, x):
		self.points.append(x)

	# calcul des points nadir et ideaux 
	def updateIdealNadir(self, y):
		node = self
		while node is not None and (np.any(y < node.pn) or np.any(y > node.pi)):
			node.pi = np.maximum(node.pi, y)
			node.pn = np.minimum(node.pn, y)
			node = node.pere            

	def updateNode(self, tree, y):
		"""
		return False if y is dominated, else True
		"""
		if np.all(self.pn >= y[1]): return False
		elif np.all(y[1] >= self.pi): 
			# Remove self and sub_tree of n
			if self.pere is not None: self.pere.remove(self)
			return True
		elif np.all(self.pi >= y[1]) or np.all(y[1] >= self.pn):
			if self.isLeaf(): 
				# on supprime les solutions dominées
				# sinon, le candidat y s'il est dominé
				L = []
				for z in self.points:
					if np.all(z[1] >= y[1]): return False
					elif np.any(z[1] > y[1]): L.append(z)
				self.points = L
			else: 
				L = []
				for x in self.points:
					if not x.updateNode(tree, y): # x domine y
						return False
					elif not x.isEmpty(): L.append(x)
				self.points = L
				# supprime les noeuds à un successeur
				if len(self.points) == 1:
					node = self.points[0]
					if self.pere is not None: self.pere.replace(self,node)
					else: tree.root = node
		self.refresh()
		return True
	
	def split(self, nChild=2):
		points = np.array([p[1] for p in self.points])
		D = [
			np.linalg.norm(points - p, axis=1).mean()
			for p in points
		]
		I = np.argsort(D)[-nChild:]
		I.sort()
		N = []
		# Create New Nodes
		for i in I[::-1]:
			z = self.points.pop(i)
			N.append(Node(z, pere=self))
		self.points, P = N, self.points
		while P: # Assign remaining solutions to leafs
			z = P.pop()
			tmp = self.closest(z[1])
			tmp.points.append(z)
			tmp.updateIdealNadir(z[1])
	
	def insert(self, y, NBMAX, nChild):
		node = self
		while not node.isLeaf(): 
			node = node.closest(y[1])
		node.points.append(y)
		node.updateIdealNadir(np.array(y[1]))
		if len(node.points) > NBMAX:
			node.split(nChild)


class NDTree():
	def __init__(self, NBMAX=20):
		self.root = None
		self.NBMAX = NBMAX

	# return (YND, Squares)
	# YND non dominated solutions
	# Squares: (depth, ideal point, nadir point) of each node
	def getPoints(self):
		def get(node):
			if node.isLeaf(): return node.points
			else:
				L = []
				for x in node.points: 
					r = get(x)
					L.extend(r)
				return L
		if self.root is None: return []
		return get(self.root)
	
	def getSquares(self, leafOnly = True):
		def get(node,count=0):
			if node.isLeaf(): return [(count,node.pi, node.pn)]
			else:
				M = [] if leafOnly else [(count,node.pi, node.pn)]
				for x in node.points: 
					yid = get(x, count+1)
					M.extend(yid)
				return M
		return get(self.root)

	def update(self,y):
		dim = len(y[1])
		if self.root is None:
			self.root = Node(y)
			return True
		elif self.root.updateNode(self, y):
			self.root.insert(y, self.NBMAX, nChild=dim+1)
			return True
		return False

	def reset(self):
		self.root = None




# ======================== Model ================================
# MR: max regret: x in O
# PMR: pairwise max regret, (x,y) in O
# MMR: minimax regret: x

# CSS:
# x: minimax regret
# y: argmax PMR(x,y)

from functools import partial
from multiprocessing import Process, Pool
import multiprocessing.managers


class MyManager(multiprocessing.managers.BaseManager):
	pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

class Model():
	def __init__(self, dim, mode='EU'):
		self.dim = dim
		self.f_normalize = 0
		self.mode = mode

		# gurobi
		self.model = gp.Model()
		self.model.Params.LogToConsole = 0

		if mode == 'EU':
			print('Init EU Model')
			self.init_eu_model()
		elif mode == 'OWA':
			print('Init OWA Model')
			self.init_owa_model()
		elif mode == 'Choquet':
			print('Init Choquet Model')
			self.init_choquet_model()
		else:
			raise Exception('Error Init Model mode=(EU,OWA,Choquet)')

	def CSS(self, X):
		"""
		return:
		x: argmin MR
		y: argmax PMR(x)
		minimax regret: MMR
		PMR(x): max PMR(x)
		"""
		PMR = self.compute_PMR(X)
		#print(PMR)
		MR = np.max(PMR,1)
		i = np.argmin(MR)
		j = np.argmax(PMR[i])
		if self.f_normalize == 0: self.f_normalize = MR.max()
		return X[i],X[j], MR[i]/self.f_normalize

	def compute_PMR(self, X, x=None):
		if self.mode == 'Choquet':  # optimization to pre-process X for Choquet
			x = (x[0], compute_xbar(x[1])) if x is not None else None 
			X = [(y[0], compute_xbar(y[1])) for y in X]
		elif self.mode == 'OWA':  # optimization to pre-process X for Choquet
			x = np.sort(x) if x is not None else None 
			X = [(y[0], np.sort(y[1])) for y in X]
		if x is None:
			return [[self.optimize(y,x) for y in X] for x in X]
		else:
			return [self.optimize(y,x) for y in X]
	
	def optimize(self, a, b):
		a, b = a[1], b[1]
		if np.array_equal(a, b): return 0
		if self.mode == 'EU': return self.optimize_eu(a, b)
		elif self.mode == 'OWA': return self.optimize_owa(a, b)
		elif self.mode == 'Choquet': return self.optimize_choquet(a, b) 
		else:
			raise 'Error optimize model, mode=(EU,OWA,Choquet)'
		
	def update(self, a, b):
		a, b = a[1], b[1]
		if self.mode == 'EU': self.update_eu(a, b)
		elif self.mode == 'OWA': self.update_owa(a, b)
		elif self.mode == 'Choquet': self.update_choquet(a, b) 
		else:
			print('Error update model, mode=(EU,OWA,Choquet)')
		self.model.update()

	# ====== EU aggregator ======
	def init_eu_model(self):
		m = self.model
		self.w = np.array([m.addVar() for _ in range(self.dim)])
		m.addConstr(sum(self.w) == 1) # may be useless

	def update_eu(self, a,b):
		self.model.addConstr(sum((a-b)*self.w) >= 0)

	# Optimize with the OWA function
	def optimize_eu(self, a, b=None):
		if b is None: b = np.zeros(self.dim)
		a, b = np.asarray(a), np.asarray(b)
		self.model.setObjective(sum(self.w*(a-b)), GRB.MAXIMIZE)
		self.model.update()
		self.model.optimize()
		return self.model.ObjVal

	# ====== OWA aggregator ======
	def init_owa_model(self):
		m = self.model
		w = np.array([m.addVar() for _ in range(self.dim)])
		self.w = w
		for i in range(self.dim-1):
			m.addConstr(w[i]-w[i+1]>=0, f'c{i+1}')
		m.addConstr(sum(w) == 1)

	def update_owa(self, a, b):
		a, b = np.sort(a), np.sort(b)
		self.update_eu(a,b)

	# Optimize with the OWA function
	def optimize_owa(self, a, b=None):
		if b is None: b = np.zeros(self.dim)
		if not np.all(a[:-1] <= a[1:]): a.sort()
		if not np.all(b[:-1] <= b[1:]): b.sort()
		return self.optimize_eu(a,b)

	# ====== Choquet aggregator ======
	# we use formulation with mass function
	def init_choquet_model(self):
		m = self.model
		self.w = np.array([m.addVar() for _ in range((1<<self.dim))])
		for x in self.w: m.addConstr(x >= 0)
		m.addConstr(sum(self.w) == 1)
		m.addConstr(self.w[0] == 0)

	def update_choquet(self, a, b):
		a, b = compute_xbar(a), compute_xbar(b)
		self.model.addConstr(sum((a-b)*self.w) >= 0)

	# Optimize with the Choquet function
	def optimize_choquet(self, a, b=None):
		if b is None: b = np.zeros(self.dim)
		#a, b = compute_xbar(a), compute_xbar(b)
		self.model.setObjective(sum(self.w*(a-b)), GRB.MAXIMIZE)
		self.model.update()
		self.model.optimize()
		return self.model.ObjVal

def eu(y, alpha):
	"""
	Compute the Weighted Sum (EU)
	y: solution in utility space
	alpha: utility weight 
	"""
	y = np.asarray(y)
	return (y*alpha).sum()

def owa(y, alpha):
	"""
	Compute the Ordered Weighted Averages (OWA)
	y: solution in utility space
	alpha: utility weight 
	"""
	return eu(np.sort(y), alpha)

def choquet(y, alpha):
	"""
	Compute the Choquet Integral
	y: solution in utility space
	alpha: mass function 
	"""
	if len(y) != len(alpha): y = compute_xbar(y)
	return (y*alpha).sum()

def compute_xbar(x):
	"""
	# compute xbar
	xbar[B] = min {x|x in B}
	"""
	dim = len(x)
	xbar = [0]*((1<<dim))
	xbar[0] = max(x)
	Lmask = [0]   
	for _ in range(dim):
		new_masks = set()
		for S in Lmask:
			for i in range(dim):
				elt = 1 << i
				# object i in S
				if S & elt: continue
				xbar[S|elt] = min(xbar[S],x[i])
				#print(bin(S|elt), bin(S)[2:], bin(elt)[2:])
				new_masks.add(S|elt)
		Lmask = new_masks
	xbar[0] = 0
	return np.array(xbar)

def generate_params(dim, mode='EU'):
	if mode == 'EU':
		alpha = np.random.dirichlet(np.ones(dim),size=1)[0]
		func_aggreg = eu
	elif mode == 'OWA':
		alpha = np.random.dirichlet(np.ones(dim),size=1)[0]
		alpha = np.sort(alpha)[::-1]
		func_aggreg = owa
	elif mode == 'Choquet':
		alpha = np.random.random(1<<dim)
		alpha[0] = 0
		alpha/=alpha.sum()
		func_aggreg = choquet
	return alpha, func_aggreg


def get_opt_eu(params, alpha):
	(n,p,v,w,W) = params
	model = gp.Model()
	model.Params.LogToConsole = 0

	obj = np.array([model.addVar(vtype=GRB.BINARY) for _ in range(n)])
	model.setObjective(sum(v@alpha*obj), GRB.MAXIMIZE)

	model.addConstr(sum(obj*w) <= W)
	model.update()
	model.optimize()
	#obj = [1<<i for i,o in enumerate(obj) if o.X > 0]
	#o = reduce(lambda a,b: a|b ,obj)
	obj = np.array([o.X for o in obj])
	return (obj, model.ObjVal)

def get_opt_owa(params, alpha):
	(n,p,v,w,W) = params
	model = gp.Model()
	model.Params.LogToConsole = 0
	poids = [p for p in alpha] + [0]
	poids = [p - poids[i+1] for i,p in enumerate(poids[:-1])]

	r = np.array([model.addVar(vtype=GRB.CONTINUOUS) for _ in range(p)])
	b = np.array([[model.addVar() for _ in range(p)] for _ in range(p)])
	obj = np.array([model.addVar(vtype=GRB.BINARY) for _ in range(n)])

	for i in range(p):
		yk = sum(v[:,i]*obj)
		for k in range(p):
			model.addConstr(r[k] - b[i,k] <= yk)
	model.addConstr(sum(obj*w) <= W)
	model.setObjective(sum(poids*(np.arange(1,p+1)*r-b.sum(0))), GRB.MAXIMIZE)

	model.update()
	model.optimize()
	obj = np.array([o.X for o in obj])
	return (obj, model.ObjVal)

def get_opt_choquet(params, alpha):
	(n,p,v,w,W) = params
	model = gp.Model()
	model.Params.LogToConsole = 0

	y = np.array([model.addVar(vtype=GRB.CONTINUOUS) for _ in range(1<<p)])
	obj = np.array([model.addVar(vtype=GRB.BINARY) for _ in range(n)])
	print(len(y))

	for A in range(1<<p):
		for j in range(p):
			if A & 1<<j == 0: continue
			model.addConstr(y[A] <= sum(obj*v[:,j]))
	model.addConstr(sum(obj*w) <= W)
	model.setObjective(sum(alpha*y), GRB.MAXIMIZE)

	model.update()
	model.optimize()
	obj = np.array([o.X for o in obj])
	return (obj, model.ObjVal)


# ===
# ===================== Regret-Based Local Search ================================
def computeInitialSolution(params):
	(n,p,v,w,W) = params
	poids = 0
	x = np.zeros(n)
	for i in np.argsort(v.mean(1))[::-1]:
		if poids + w[i] <= W:
			poids += w[i]
			x[i] = 1
	value = v[x!=0,:].sum(0)
	return (x, value)

class DecisionMaker():
	def __init__(self, dim, mode='EU', alpha = None):
		self.params, self.func_aggreg = generate_params(dim, mode=mode)
		self.mode = mode
		self.nb_questions = 0
		if alpha is not None: self.params = alpha

	def ask(self, a, b):
		self.nb_questions += 1
		a =  self.func_aggreg(a[1],self.params)
		b = self.func_aggreg(b[1], self.params)
		return a >= b
	
	def value(self, x):
		return self.func_aggreg(x[1], self.params)
	
	def get_opt(self, params):
		if self.mode == 'EU': return get_opt_eu(params, self.params)
		if self.mode == 'OWA': return get_opt_owa(params, self.params)
		if self.mode == 'EU': return get_opt_choquet(params, self.params)
		
def RBLS(params, mode='EU', P=[], eps=1e-3, max_it=200, DM = None, NBMAX=20, verbose=False):
	# Initialization
	p = params[1]
	if DM is None: DM = DecisionMaker(p, mode=mode)
	model = Model(dim=p, mode=mode)
	for a,b in P: model.update(a, b)
	x_star = computeInitialSolution(params)
	it = 0
	improve = True
	#  Local Search
	while improve and (it < max_it):
		# Generation of neighbors
		Xe = NDTree(NBMAX)
		Xe.update(x_star)
		voisins = voisinage(x_star, params)
		if verbose:
			print(f'{it = } | voisins: {len(voisins)}')
		for voisin in voisins: 
			if np.all(x_star[1] >= voisin[1]): continue
			Xe.update(voisin)
		X = Xe.getPoints()
		# regret-based elicitation:
		o1, o2, regret = model.CSS(X)
		while regret > eps:
			if DM.ask(o1, o2): model.update(o1,o2)
			else: model.update(o2,o1)
			o1, o2, regret = model.CSS(X)
		PMR = model.compute_PMR(X, x_star)
		improve = max(PMR) > eps and o1 is not x_star
		if improve:
			x_star = o1
			if verbose:
				print(f'solution: {max(PMR)/model.f_normalize}\n{x_star[1]}')
				print()
			it += 1
	return x_star

# ===================== Regret-Based Global Search ================================
def RBGS(m, params, mode='EU', eps=1e-3, max_it=200, DM = None, NBMAX=20, verbose=False):
	# Phase I: PLS
	X = PLS(m,params, NBMAX)

	# Phase II: Elicitation 
	model = Model(dim=params[1], mode=mode)
	o1, o2, regret = model.CSS(X)
	minmax_regrets = [regret]
	it = 1
	while regret > eps and it < max_it:
		if DM.ask(o1, o2): model.update(o1,o2)
		else: model.update(o2,o1)
		o1, o2, regret = model.CSS(X)
		minmax_regrets.append(regret)
		if verbose: print(f'{it = } | {regret =: .2f} | {o1[1]} {o2[1]}')
		it += 1
	return o1
