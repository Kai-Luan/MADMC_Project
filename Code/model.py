import numpy as np
import gurobipy as gp
from gurobipy import GRB
from functools import partial
from multiprocessing import Process, Pool
import multiprocessing.managers

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


def fill_matrix(args, matrix_to_fill, solutions, func, ):
	i, j = args
	matrix_to_fill[i,j] = func(solutions[j][1], solutions[i][1])

def compute_PMR(X, dim, P):
    """
    X: ensemble des solutions (sac, criteres)
        x: solution courante (x*)
    """
    n = len(X)
    manager = MyManager()
    manager.start()
    PMR = manager.np_zeros((n,n))
    pool = Pool()
    run_list = [(i,j) for i in range(n) for j in range(n) if i!=j]
    local_func = partial(optimize_eu, dim = dim, P = P)
    fun = partial(fill_matrix, func=local_func, solutions = X, matrix_to_fill = PMR)
    pool.map(fun, run_list)
    return PMR

def optimize_eu(a, b, dim, P=[]):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
		
        m = gp.Model(env = env)
        m.Params.OutputFlag = 0
        w = np.array([m.addVar() for _ in range(dim)])
        m.addConstr(sum(w) == 1) # may be useless
        for x in P: m.addConstr(sum(x*w) >= 0) # <- w.x >= w.y
        if b is None: b = np.zeros(dim)
        a, b = np.asarray(a), np.asarray(b)
        m.setObjective(sum(w*(a-b)), GRB.MAXIMIZE) # <- PMRws(x,y;P) = maxw [y.w - x.w]
        m.update()
        m.optimize()
        return m.ObjVal
	

class Model_MultiProcess():
	def __init__(self, dim, mode='EU'):
		self.dim = dim
		self.f_normalize = 0
		self.mode = mode

		# Memory choquet
		self.memory = dict()
		self.P = []
		
	def init_model(self):
		mode = self.modes
		if mode == 'EU':
			print('Init EU Model')
			return self.init_eu_model()
		else:
			raise Exception('Error Init Model mode=(EU,OWA,Choquet)')
	
    #Current solution strategy
	def CSS(self, X):
		"""
  		X: base des solutions possibles (criteres) 
		return:
		x: argmin MR
		y: argmax PMR(x)
		minimax regret: MMR
		PMR(x): max PMR(x)
		"""
		# resolution d'un PL
		# Recupere toutes les paires de PMR possibles
		PMR = self.compute_PMR(X)
		#print(PMR)
		#On prend le max regret pour chaque x (pour un x donnée, on a |X|-1 PMR)
		MR = np.max(PMR,1)
		# On prend celui qui minimise le max regret (y)
		i = np.argmin(MR)
		# indice correspondant a l'ensemble dans les PMR (x)
		j = np.argmax(PMR[i])
		# Facteur de normalisation puis on renvoie x, y selon la formule de l'article et on normalise la valeur du minmax regret
		if self.f_normalize == 0: self.f_normalize = MR.max()
		return X[i],X[j], MR[i]/self.f_normalize

	def compute_PMR(self, X, x=None):
		"""
  		X: ensemble des solutions (sac, criteres)
    		x: solution courante (x*)
  		"""
		return compute_PMR(X, self.dim, self.P)
	
	def update(self, a, b):
		a, b = a[1], b[1]
		if self.mode == 'EU': self.update_eu(a, b)
		elif self.mode == 'OWA': self.update_owa(a, b)
		elif self.mode == 'Choquet': self.update_choquet(a, b) 
		else:
			print('Error update model, mode=(EU,OWA,Choquet)')

	# ====== EU aggregator or Weighted Sum ======
	def update_eu(self, a,b):
		self.P.append((a-b))
