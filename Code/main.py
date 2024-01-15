import numpy as np
import matplotlib.pyplot as plt
from utils import *
from model import *
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
    """
    Lance une experience sur la première méthode de résolution et le sauvegarde dans un fichier csv
    avec le nombre de questions posée et variation du regret minimax selon le nombre de questions posées
    paramèters:
    - params: problème du sac à dos (n,p,v,w,W)
            - n: nombre d'objets
            - p: nombre de critères
            - v: valeurs des objets sur p critères
            - w: poids des objets
            - W: capacité du sac à dos
    - output_file: nom du fichier csv avec les données de l'expérimentation
    - mode: le nom de la fonction d'agrégation
    - nb_run: le nombre de simulation
    """
    print('======= Experience 1 ========')
    nb_initial = 20
    verbose = True
    NBMAX = 20
    print('Generation des solutions non-dominatées  ...')
    YND = PLS(nb_initial, params, NBMAX,verbose= False)
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
    P.to_csv(output_file, mode='w')
    print('Finished')

def experience2(params, output_file, mode = 'EU', nb_run = 20):
    """
    Lance une experience et le sauvegarde dans un fichier csv
    On compare la 1e et la 2e méthode selon:
        - le temps de calcul
        - l'erreur par rapport à la solution optimale du décideur
        - le nombre de questions posées
    paramèters:
    - params: problème du sac à dos (n,p,v,w,W)
            - n: nombre d'objets
            - p: nombre de critères
            - v: valeurs des objets sur p critères
            - w: poids des objets
            - W: capacité du sac à dos
    - output_file: nom du fichier csv avec les données de l'expérimentation
    - mode: le nom de la fonction d'agrégation
    - nb_run: le nombre de simulation
    """
    print('===== Experience 2 ========')
    P = []
    nb_initial = 20
    for it in range(nb_run):
        print(f'======= {it = } / {nb_run-1} ========')
        res1, res2 = step(params, mode, nb_initial, verbose=True)
        P.append(res1)
        P.append(res2)
    P = np.array(P) # (nb queries, gap, time)
    labels = ['Methode', 'Number of queries', 'Gap', 'Times']
    P = pd.DataFrame(P, columns=labels)
    P.to_csv(output_file, mode='w')
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
    n = 60
    # nombre de critères
    p = 3
    w = w[:n] # poids des objets
    v = v[:n,:p] # valeurs des objets sur les p critèress
    W = w.sum()//2 # capacité du sac à dos
    params = (n,p,v,w,W)

    experience1(params=params, output_file=f'data/{mode}/exp1_{p}KP{n}.log', mode=mode, nb_run=20)
    #experience2(params=params, output_file=f'data/{mode}/exp2_{p}KP{n}.log', mode=mode, nb_run=20)