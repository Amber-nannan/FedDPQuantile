from util_fdp import *
import ray

"""
main code 同分布，但响应可能不同
"""

def generate_lists(start, end, K):
    list_start = [start] * K
    list_avg = np.linspace(start, end, K).tolist()
    list_end = [end] * K
    return list_start, list_avg, list_end

ray.init(runtime_env={"working_dir": "."})  # 设置工作目录
                    #   "py_modules": [".."]

dist_type = 'normal'   # types = ['normal', 'uniform', 'cauchy']
# gene_process = 'homo'
gene_process = 'hete'
mode='federated'
n_sim = 1000
seed = 2025

Ts = [5000,50000]
# Ts = [5000]
taus = [0.3,0.5,0.8]
# taus = [0.5]
# rs = [0.25,0.9]
rs = [0.25]

n_clients = 10
client_rss = [[rs[i]]*n_clients for i in range(len(rs))]
rs_names = rs
# client_rss = generate_lists(rs[0], rs[1], n_clients)
# rs_names = [rs[0],rs[1]]

Es = [1, 5,'log']
nn_ct = len(Ts)*len(taus)*len(client_rss)*len(Es)

# 初始化结果存储字典（使用defaultdict自动创建嵌套结构）
cvgdict = {}
maedict = {}

# key=(r,E), value=(a,b,c)
abc_dict = {
    (0.25, 1):(0.501, 0, 2), 
    (0.25, 5):(0.546, 0, 1), 
    (0.25, 'log'):(0.546, 0, 1), 
    (0.9, 1): (0.568, 0, 1),
    (0.9, 5): (0.568, 0, 1),
    (0.9, 'log'): (0.782, 25, 2),
}

# 联邦模拟
ct = 0
for T in Ts:
    # 初始化当前样本量的字典层级
    cvgdict[T] = {}
    maedict[T] = {}
    for tau in taus:
        # 初始化当前分位数的字典层级
        cvgdict[T][tau] = {}
        maedict[T][tau] = {}
        for name_idx, client_rs in enumerate(client_rss):
            name = rs_names[name_idx]

            if name == 'hetero':   # table2 暂时不关注hetero
                continue

            cvgdict[T][tau][name] = {}
            maedict[T][tau][name] = {}
            for E in Es:
                E_typ = 'log' if E == 'log' else 'cons'
                a, b, c = abc_dict[(name, E)]  # 不同r(names)、E的组合对应不同的 (a,b,c)

                t1 = time.time()
                fed_results = run_federated_simulation(
                    dist_type=dist_type,tau=tau,
                    client_rs=client_rs,n_clients=n_clients,
                    T=T,E_typ=E_typ,E_cons=E,gene_process=gene_process,
                    mode=mode,
                    n_sim=n_sim,base_seed=seed,
                    a=a, b=b,c=c)
                # 分析结果
                z_score = 6.753 if E == 'log' else 6.74735
                output = analyze_results(fed_results,z_score=z_score)
                cvg = output['coverage']
                mae = output['mae']

                # 存储结果
                cvgdict[T][tau][name][E] = cvg
                maedict[T][tau][name][E] = mae
                t2 = time.time()
                ct += 1
                save_pickle(cvgdict, f'./case_rs_{mode}_{gene_process}_cvg.pkl')
                save_pickle(maedict, f'./case_rs_{mode}_{gene_process}_mae.pkl')
                print(f'Ts:{T} tau:{tau} name:{name} E:{E} TC:{(t2-t1)/60:.2f}min LTC:{(t2-t1)*(nn_ct-ct)/60:.2f}min')
                
ray.shutdown()
