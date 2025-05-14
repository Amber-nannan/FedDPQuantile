from util_fdp import *
import ray


"""
main code 分布和响应r均异质，此时我们只能考虑中位数
"""

def generate_lists(start, end, K):
    list_start = [start] * K
    list_avg = np.linspace(start, end, K).tolist()
    list_end = [end] * K
    return list_start, list_avg, list_end

ray.init(runtime_env={"working_dir": "."})  # 设置工作目录


gene_process = 'hete' # 分布和响应r均异质
mode='federated'
n_sim = 1000
seed = 2025

Ts = [10000, 50000]
# Ts = [5000]
# taus = [0.3,0.5,0.8]
taus = [0.5]
rs = [0.25,0.9]

## Case 2 n clients=10, E=1,5,log

n_clients = 10
client_rss = generate_lists(rs[0], rs[1], n_clients)
rs_names = [rs[0],'hetero',rs[1]]
# client_rss = [generate_lists(rs[0], rs[0], n_clients)[0]]
# rs_names = [rs[0]]

Es = [1,5,'log']
nn_ct = len(Ts)*len(taus)*len(client_rss)*len(Es)

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
            for E in Es:
                E_typ = 'log' if E == 'log' else 'cons'
                t1 = time.time()
                fed_results = run_federated_simulation(
                    dist_type=None,tau=tau,
                    client_rs=client_rs,n_clients=n_clients,
                    T=T,E_typ=E_typ,E_cons=E,gene_process=gene_process,
                    mode=mode,
                    n_sim=n_sim,base_seed=seed,a=0.51, b=100,c=20)
                # 分析结果
                z_score = 6.753 if E == 'log' else 6.74735
                output = analyze_results(fed_results,z_score=z_score)
                cvg = output['coverage'];mae = output['mae']

                # 存储结果
                cvgdict[T][tau][name][E] = cvg;maedict[T][tau][name][E] = mae
                t2 = time.time()
                ct += 1
                save_pickle(cvgdict, f'./case_2_{mode}_{gene_process}_cvg.pkl')
                save_pickle(maedict, f'./case_2_{mode}_{gene_process}_mae.pkl')
                print(f'Ts:{T} tau:{tau} name:{name} E:{E} TC:{(t2-t1)/60:.2f}min LTC:{(t2-t1)*(nn_ct-ct)/60:.2f}min')
                
ray.shutdown()