from util_fdp import *
import ray

"""
Main code: Global results for same distribution family with different means
"""

ray.init(runtime_env={"working_dir": "."})  # Set working directory

os.makedirs("output", exist_ok=True)
dist_type = 'normal'   # types = ['normal', 'uniform', 'cauchy']
gene_process = 'hete'
mode='global'
T_mode='samples'  # samples / rounds
n_sim = 1000
seed = 2025
n_clients = 10


Ts = [10000, 50000] if T_mode == 'samples' else [5000,50000]
taus = [0.3,0.5,0.8]
rs = [0.25,0.9]

n_clients = 10
client_rss = [[rs[i]]*n_clients for i in range(len(rs))]
rs_names = rs

Es = [1, 5,'log']
nn_ct = len(Ts)*len(taus)*len(client_rss)*len(Es)

# Initialize result storage dictionaries
cvgdict = {}
maedict = {}


abc_dict = {
    0.25:(0.51, 100, 2), 
    0.25:(0.51, 100, 2), 
    0.25:(0.51, 100, 2), 
    0.9: (0.51, 100, 2),
    0.9: (0.51, 100, 2),
    0.9: (0.51, 100, 2),
}

# Federated simulation
ct = 0
for T in Ts:
    # Initialize dictionary level for current T
    cvgdict[T] = {};maedict[T] = {}
    for tau in taus:
        # Initialize dictionary level for current quantiles
        cvgdict[T][tau] = {};maedict[T][tau] = {}
        for name_idx, client_rs in enumerate(client_rss):
            name = rs_names[name_idx]
            cvgdict[T][tau][name] = {};maedict[T][tau][name] = {}
            a, b, c = abc_dict[name]
            t1 = time.time()
            fed_results = run_federated_simulation(
                dist_type=dist_type,taus=tau,
                client_rs=client_rs,n_clients=n_clients,
                T=T,E_typ='cons',E_cons=1,gene_process=gene_process,mode=mode,T_mode=T_mode,
                n_sim=n_sim,base_seed=seed,a=a, b=b,c=c)
            # Analyze results
            output = analyze_results(fed_results,z_score=6.74735)
            cvg = output['coverage'];mae = output['mae']
    
            # Store results 
            cvgdict[T][tau][name] = cvg;maedict[T][tau][name] = mae
            t2 = time.time()
            ct += 1
            save_pickle(cvgdict, f'output/case_{mode}_{gene_process}_cvg.pkl');save_pickle(maedict, f'output/case_{mode}_{gene_process}_mae.pkl')
            print(f'Ts:{T} tau:{tau} TC:{(t2-t1)/60:.2f}min LTC:{(t2-t1)*(nn_ct-ct)/60:.2f}min')
                    
ray.shutdown()

