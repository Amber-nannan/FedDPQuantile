from util_fdp import *
import ray

"""
main code 同分布，但响应可能不同
"""

ray.init(runtime_env={"working_dir": "."})  # 设置工作目录
os.makedirs("output", exist_ok=True)

dist_type = 'normal'   # types = ['normal', 'uniform', 'cauchy']
gene_process = 'homo' # 'hete' / 'hete_d'
mode='federated' # federated / global
T_mode='samples'  # samples / rounds
n_sim = 100
seed = 42

Ts = [10000, 50000] if T_mode == 'samples' else [5000,50000]
tauss = [[0.5]*10,np.linspace(0.3, 0.5, 10).tolist(),
        np.linspace(0.5, 0.8, 10).tolist()]
rs = [0.25,0.9]

n_clients = 10
client_rss = generate_lists(rs[0], rs[1], n_clients)
rs_names = [rs[0],'hetero',rs[1]]

# Es = ['log']
Es = [1, 5,'log',1/2]
nn_ct = len(Ts)*len(tauss)*len(client_rss)*len(Es)

# 初始化结果存储字典（使用defaultdict自动创建嵌套结构）
cvgdict = {}
maedict = {}

Z_SCORES = {
    'log': 6.753,
    'cons': 6.74735,
     1/3: 6.339,        
     1/2: 5.851,        
     2/3: 4.993
}



# 联邦模拟
ct = 0
for T in Ts:
    # 初始化当前样本量的字典层级
    cvgdict[T] = {};maedict[T] = {}
    for i,taus in enumerate(tauss):
        # 初始化当前分位数的字典层级
        cvgdict[T][i] = {};maedict[T][i] = {}
        for name_idx, client_rs in enumerate(client_rss):
            name = rs_names[name_idx]
            cvgdict[T][i][name] = {};maedict[T][i][name] = {}
            for E in Es:
                E_typ = 'log' if E == 'log' else ('cons' if isinstance(E, int) else E)
                t1 = time.time()
                fed_results = run_federated_simulation(
                    dist_type=dist_type,taus=taus,
                    client_rs=client_rs,n_clients=n_clients,
                    T=T,E_typ=E_typ,E_cons=E,gene_process=gene_process,
                    mode=mode,
                    n_sim=n_sim,base_seed=seed,a=0.51, b=100,c=20,
                T_mode=T_mode)
                # 分析结果
                z_score = Z_SCORES[E_typ] 
                output = analyze_results(fed_results,z_score=z_score)
                cvg = output['coverage'];mae = output['mae']

                # 存储结果
                cvgdict[T][i][name][E] = cvg;maedict[T][i][name][E] = mae
                t2 = time.time()
                ct += 1
                save_pickle(cvgdict, f'output/test_{dist_type}_{mode}_{T_mode}_{gene_process}_cvg.pkl')
                save_pickle(maedict, f'output/test_{dist_type}_{mode}_{T_mode}_{gene_process}_mae.pkl')
                print(f'Ts:{T} taus:{taus} name:{name} E:{E} TC:{(t2-t1)/60:.2f}min LTC:{(t2-t1)*(nn_ct-ct)/60:.2f}min')
                
ray.shutdown()