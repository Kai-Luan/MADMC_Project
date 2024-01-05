import numpy as np
import matplotlib.pyplot as plt
from utils import *
from read_file import *
from time import time
import pandas as pd

def gap(opt, sol):
    return (opt - sol)*100 / opt

def value(x, v):
    print(x.shape)
    return v[x==1, :].sum(0)

def step(params, mode='EU', m = 20, verbose=False):
    DM = DecisionMaker(p, mode)
    DM_opt = DM.get_opt(params)
    print('Optimal: ', value(DM_opt[0], params[2]))

    # Procedure 1
    DM.nb_questions
    print('Procedure 1')
    tps = time()
    opt = RBGS(m, params, mode=mode, DM=DM, verbose=verbose)
    tps = time() - tps
    n = DM.nb_questions
    g = gap(DM_opt[1], DM.value(opt))
    res1 = (1, n, g, tps)
    
    DM.nb_questions = 0

    #Procedure 2
    print('Procedure 2')
    tps = time()
    opt = RBLS(params, mode=mode, DM=DM, verbose=verbose)
    tps = time() - tps
    n = DM.nb_questions
    g = gap(DM_opt[1], DM.value(opt))
    res2 = (2, n, g, tps)
    DM.nb_questions = 0
    return res1, res2

def experience1(params, output_file, mode='EU', nb_run=20):
    print('Experience 1')
    m = 20
    verbose = True
    NBMAX = 20
    print('Genereting non-dominated solutions ...')
    YND = PLS(m, params, NBMAX,verbose= False)
    points = list(map(lambda x: x[1], YND))
    points = np.array(points)
    print(f'nombre de points non-dominés trouvés: {len(points)}')

    eps = 1e-4
    L = []
    it_max = 50
    for epoch in range(nb_run):
        print(f'{epoch = } / {nb_run-1}')
        DM = DecisionMaker(p, mode)
        model = Model(dim=p, mode=mode)
        o1, o2, regret = model.CSS(YND)
        minmax_regrets = [regret]
        it = 0
        while regret > eps and it < it_max:
            if DM.ask(o1, o2): model.update(o1,o2)
            else: model.update(o2,o1)
            o1, o2, regret = model.CSS(YND)
            minmax_regrets.append(regret)
        L.append(minmax_regrets)

    mean_regrets = np.zeros((nb_run,it_max))
    for i,regrets in enumerate(L):
        mean_regrets[i,0] = len(regrets)
        mean_regrets[i,1:len(regrets)+1] = regrets
    P = pd.DataFrame(mean_regrets)
    P.to_csv(output_file, mode='a')
    print('Finish')

def experience2(params, output_file, mode = 'EU', nb_run = 20):
    print('===== Experience 2 ========')
    P = []

    m = 20
    for it in range(nb_run):
        print(f'======= {it = } / {nb_run-1} ========')
        res1, res2 = step(params, mode, m, verbose=True)
        P.append(res1)
        P.append(res2)
    P = np.array(P) # (nb queries, gap, time)
    labels = ['Methode', 'Number of queries', 'Gap', 'Times']
    P = pd.DataFrame(P, columns=labels)
    P.to_csv(output_file, mode='a')
    print('Finished')
 
if __name__ == '__main__':
    ############ NE PAS MODEIFIER ####################
    ## Loading data
    n = 200
    p = 6
    filename = f"./data/2KP200-TA-0.dat"
    w=np.zeros(n,dtype=int) # poids des objets
    v=np.zeros((n,p),dtype=int) # utilités des objets
    W = readFile(filename,w,v)
    #################################################
    # ====== On prend un sous-ensemble du problème ====
    mode = 'EU'
    # nombre d'objets
    n = 20
    # nombre de critères
    p = 3
    w = w[:n] # poids des objets
    v = v[:n,:p] # valeurs des objets sur les p critèress
    W = w.sum()//2 # capacité du sac à dos
    params = (n,p,v,w,W)

    experience1(params=params, output_file=f'data/{mode}/exp1_{p}KP{n}.csv', mode=mode, nb_run=20)
    experience2(params=params, output_file=f'data/{mode}/exp2_{p}KP{n}.csv', mode=mode, nb_run=20)