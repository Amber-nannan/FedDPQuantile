from util_fdp import *
import ray

"""
main code 全局结果
"""

def generate_lists(start, end, K):
    list_start = [start] * K
    list_avg = np.linspace(start, end, K).tolist()
    list_end = [end] * K
    return list_start, list_avg, list_end

ray.init(runtime_env={"working_dir": "."})  # 设置工作目录



dist_type = 'normal'   # types = ['normal', 'uniform', 'cauchy']
gene_process = 'hete'
mode='global'
n_sim = 1000
seed = 2025
n_clients = 10

Ts = [5000,50000]   # communication rounds
taus = [0.3,0.5,0.8]

client_rs = [1.0]*n_clients if mode == 'federated' else [1.0]

nn_ct = len(Ts)*len(taus)

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
        t1 = time.time()
        fed_results = run_federated_simulation(
            dist_type=dist_type,tau=tau,
            client_rs=client_rs,n_clients=n_clients,
            T=T,E_typ='cons',E_cons=1,gene_process=gene_process,mode=mode,
            n_sim=n_sim,base_seed=seed)
        # 分析结果
        output = analyze_results(fed_results,z_score=6.74735)
        cvg = output['coverage'];mae = output['mae']

        # 存储结果
        cvgdict[T][tau] = cvg;maedict[T][tau] = mae
        t2 = time.time()
        ct += 1
        save_pickle(cvgdict, f'./case_{mode}_{gene_process}_cvg.pkl');save_pickle(maedict, f'./case_{mode}_{gene_process}_mae.pkl')
        print(f'Ts:{T} tau:{tau} TC:{(t2-t1)/60:.2f}min LTC:{(t2-t1)*(nn_ct-ct)/60:.2f}min')
                
ray.shutdown()
