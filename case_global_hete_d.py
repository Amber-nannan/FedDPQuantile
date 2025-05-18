from util_fdp import *
import ray

"""
main code 全局结果
"""

ray.init(runtime_env={"working_dir": "."})  # 设置工作目录
os.makedirs("output", exist_ok=True)
dist_type = 'normal'   
gene_process = 'hete_d' # 'homo /hete' / 'hete_d'
mode='global'
T_mode='samples'  # samples / rounds
n_sim = 1000
seed = 42 if gene_process == 'homo' else 2025
n_clients = 10
a,b,c = 0.51, 100, 2

Ts = [10000, 50000] if T_mode == 'samples' else [5000,50000]
taus = [0.3,0.5,0.8] if gene_process == 'homo' else [0.5]

rs = [0.25,0.9]
client_rss = generate_lists(rs[0], rs[1], n_clients)
rs_names = [rs[0],'hetero',rs[1]]

nn_ct = len(Ts)*len(taus)*len(client_rss)

# 初始化结果存储字典（使用defaultdict自动创建嵌套结构）
cvgdict = {}
maedict = {}

# 联邦模拟
ct = 0
for T in Ts:
    # 初始化当前样本量的字典层级
    cvgdict[T] = {};maedict[T] = {}
    for tau in taus:
        # 初始化当前分位数的字典层级
        cvgdict[T][tau] = {};maedict[T][tau] = {}
        for name_idx, client_rs in enumerate(client_rss):
            name = rs_names[name_idx]
            cvgdict[T][tau][name] = {};maedict[T][tau][name] = {}
        
            t1 = time.time()
            fed_results = run_federated_simulation(
                dist_type=dist_type,taus=tau,
                client_rs=client_rs,n_clients=n_clients,
                T=T,E_typ='cons',E_cons=1,gene_process=gene_process,mode=mode,
                n_sim=n_sim,base_seed=seed,
                T_mode=T_mode, a=a, b=b,c=c)
            # 分析结果
            output = analyze_results(fed_results,z_score=6.74735)
            cvg = output['coverage'];mae = output['mae']
    
            # 存储结果
            cvgdict[T][tau][name] = cvg;maedict[T][tau][name] = mae
            t2 = time.time()
            ct += 1
            save_pickle(cvgdict, f'output/case_{mode}_{gene_process}_cvg.pkl');save_pickle(maedict, f'output/case_{mode}_{gene_process}_mae.pkl')
            print(f'Ts:{T} tau:{tau} TC:{(t2-t1)/60:.2f}min LTC:{(t2-t1)*(nn_ct-ct)/60:.2f}min')
                    
ray.shutdown()
