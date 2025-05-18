from util_fdp import *
import ray

"""
Main code: Global results for identical distribution
"""

ray.init(runtime_env={"working_dir": "."})  # Set working directory
os.makedirs("output", exist_ok=True)

dist_type = 'normal' 
gene_process = 'homo' # 'hete' / 'hete_d'
mode='global'
T_mode='samples'
n_sim = 1000
seed = 42 if gene_process == 'homo' else 2025
n_clients = 10
a,b,c = 0.51, 100, 2

Ts = [10000, 50000] if T_mode == 'samples' else [5000,50000]
tauss = [[0.5]*10,np.linspace(0.3, 0.5, 10).tolist(),
        np.linspace(0.5, 0.8, 10).tolist()]

rs = [0.25,0.9]
client_rss = generate_lists(rs[0], rs[1], n_clients)
rs_names = [rs[0],'hetero',rs[1]]

nn_ct = len(Ts)*len(tauss)*len(client_rss)

# Initialize result storage dictionaries
cvgdict = {}
maedict = {}

# Federated simulation
ct = 0
for T in Ts:
    # Initialize dictionary level for current T
    cvgdict[T] = {};maedict[T] = {}
    for i,taus in enumerate(tauss):
        # Initialize dictionary level for current quantiles
        cvgdict[T][i] = {};maedict[T][i] = {}
        for name_idx, client_rs in enumerate(client_rss):
            name = rs_names[name_idx]
            cvgdict[T][i][name] = {};maedict[T][i][name] = {}
        
            t1 = time.time()
            fed_results = run_federated_simulation(
                dist_type=dist_type,taus=taus,
                client_rs=client_rs,n_clients=n_clients,
                T=T,E_typ='cons',E_cons=1,gene_process=gene_process,mode=mode,
                n_sim=n_sim,base_seed=seed,
                T_mode=T_mode, a=a, b=b,c=c)
            # Analyze results
            output = analyze_results(fed_results,z_score=6.74735)
            cvg = output['coverage'];mae = output['mae']
    
            # Store results
            cvgdict[T][i][name] = cvg;maedict[T][i][name] = mae
            t2 = time.time()
            ct += 1
            save_pickle(cvgdict, f'output/case_{mode}_{gene_process}_cvg.pkl');save_pickle(maedict, f'output/case_{mode}_{gene_process}_mae.pkl')
            print(f'Ts:{T} taus:{taus} TC:{(t2-t1)/60:.2f}min LTC:{(t2-t1)*(nn_ct-ct)/60:.2f}min')
                    
ray.shutdown()
