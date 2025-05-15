# 使用示例
import sys
import time
sys.path.append('..')
from util import *
from FedDPQuantile import FedDPQuantile
from DPQuantile import DPQuantile
import pickle
import os
from scipy.stats import norm
from scipy.optimize import root_scalar

def objective(x, mu_list, taus):
    """全局目标函数：计算 (1/K) * sum(Φ(x - μ_k)) - τ （σ=1）"""
    return np.mean(norm.cdf(x - np.asarray(mu_list))) - np.mean(taus)    # 这里隐藏的设定：（1）每一台机器的数据都是正态分布 （2）每一台机器权重相同

def solve_quantile(mu_list, taus, xtol=1e-8):
    """求解方程：(1/K) * sum(Φ(x - μ_k)) = τ （σ=1，权重均匀）"""
    # 确定搜索区间（覆盖99.99994%概率范围）
    mu_min, mu_max = np.min(mu_list), np.max(mu_list)
    bracket = (mu_min - 5, mu_max + 5)
    # 通过lambda绑定参数调用外部目标函数
    result = root_scalar(lambda x: objective(x, mu_list, taus),
                         method='bisect',bracket=bracket,xtol=xtol)
    return result.root

def get_Em_list(T, warm_up=0.05, typ='log', E_cons=1, T_mode='rounds'):
    """
    生成每轮的本地迭代次数列表
    
    参数:
        T: 当 T_mode='samples' 时表示总样本量，当 T_mode='rounds' 时表示通信轮数
        warm_up: 预热阶段比例
        typ: 迭代次数类型，'log'或'cons'
        E_cons: 常数迭代次数（当typ='cons'时使用）
    
    返回:
        总样本量, 迭代次数列表
    """
    if T_mode == 'rounds':
        # 基于通信轮数
        pre = int(T * warm_up)
        minor = T - pre
        if typ == 'log':
            Em = [int(np.ceil(np.log2(m + 1))) for m in range(1, minor + 1)]
        elif typ == 'cons':
            Em = [E_cons] * minor
        return pre + sum(Em), [1] * pre + Em
    
    else:  # T_mode == 'samples' 基于总样本量
        total_samples = T
        if typ == 'cons':
            # 对于常数迭代次数，直接计算
            warm_up_samples = int(total_samples * warm_up)
            remaining_samples = total_samples - warm_up_samples
            
            # 计算需要多少轮常数迭代
            rounds = remaining_samples // E_cons
            leftover = remaining_samples % E_cons
            
            # 生成迭代次数列表
            Em_list = [1] * warm_up_samples + [E_cons] * rounds
            if leftover > 0:
                Em_list.append(leftover)
                
            return total_samples, Em_list
            
        elif typ == 'log':
            # 使用log₂(2), log₂(3), ..., log₂(n)的方式生成迭代次数
            warm_up_samples = int(total_samples * warm_up)
            
            # 计算预热阶段后还剩多少样本量
            remaining_samples = total_samples - warm_up_samples
            
            # 生成对数序列直到总和接近但不超过remaining_samples
            Em = []
            n = 2  # 从log₂(2)开始
            current_sum = 0
            
            while True:
                log_value = int(np.ceil(np.log2(n)))
                if current_sum + log_value >= remaining_samples:
                    break
                
                Em.append(log_value)
                current_sum += log_value
                n += 1
            
            # 特殊处理：调整最后一个元素使总和恰好等于total_samples
            if current_sum < remaining_samples:
                Em.append(remaining_samples - current_sum)
            
            # 生成最终的迭代次数列表
            Em_list = [1] * warm_up_samples + Em
            return total_samples, Em_list

def save_pickle(var, file_path):
    """保存变量到pickle文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(var, f)

def load_pickle(file_path):
    """从pickle文件加载变量"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def train(seed, dist_type, tau, client_rs, n_clients, T, E_typ='log', E_cons=1,
          gene_process='homo', mode='federated', use_true_q_init=False, a=0.51, b=100,c=2,
          return_history=False):
    """
    单次联邦实验（合并全局训练和联邦训练版本）
    
    参数:
        mode: 'federated'（联邦训练）或 'global'（全局训练）
        T_mode: 'rounds'（基于通信轮数）或 'samples'（基于总样本量）
    """
    np.random.seed(42)
    # mus = np.random.randn(n_clients) if gene_process == 'hete' else np.zeros(n_clients)
    if gene_process == 'hete':
        mus = np.random.randn(n_clients)
    elif gene_process == 'homo' or gene_process=='hete_d':
        mus = np.zeros(n_clients)
    elif isinstance(gene_process,float):
        mus = np.random.normal(loc=0,scale=gene_process,size=n_clients)
        
    clients_data = []
    ET, Em_list = get_Em_list(T, typ=E_typ, E_cons=E_cons, T_mode=T_mode)

    # if gene_process in ['homo', 'hete']:
    if gene_process != 'hete_d':
        for k in range(n_clients):
            data, _ = generate_data(dist_type, taus[k], ET, mu=mus[k])    # ET个sample
            clients_data.append(data)
        global_true_q = solve_quantile(mus, taus)

    elif gene_process == 'hete_d':
        # 三种分布类型：normal, uniform, cauchy
        distribution_pool = ['normal', 'uniform', 'cauchy']
        
        # 尽量平均地把 n_clients 划分给三种分布
        c1 = n_clients // 3
        c3 = n_clients - 2*c1
        # 得到 dist_list，例如对 10 台机器 => ['normal','normal','normal', ... 'uniform' x3, 'cauchy' x4]
        dist_list = []
        dist_list += [distribution_pool[0]] * c1
        dist_list += [distribution_pool[1]] * c1
        dist_list += [distribution_pool[2]] * c3
        
        # 为每个客户端生成对应分布的数据
        for k in range(n_clients):
            data, _ = generate_data(dist_list[k], taus[k], ET, mu=mus[k])
            clients_data.append(data)

        global_true_q = 0.0 # 只能接受中位数
    
    # 根据模式选择数据处理方式
    if mode == 'global':
        # 全局训练模式：合并数据，n_clients=1
        Q_avgs = []; Vars = []
        for i,data_i in enumerate(clients_data):
            model = DPQuantile(tau=taus[i], r=client_rs[i], true_q=global_true_q,a=a, b=b,c=c,seed=seed)
            model.fit(data_i)
            Q_avgs.append(model.Q_avg)
            Vars.append(model.get_variance())
        return global_true_q, np.mean(Q_avgs), np.mean(Vars), _

    elif mode == 'federated':
        # 联邦训练模式：保留客户端数据
        model = FedDPQuantile(n_clients=n_clients, client_rs=client_rs,
                              taus=taus,
                              true_q=global_true_q,use_true_q_init=use_true_q_init,a=a, b=b,c=c,seed=seed)
        model.fit(clients_data, Em_list)

    if return_history:
        return model.get_stats_history()
    else:
        return global_true_q, model.Q_avg, model.get_variance(), model.errors


@ray.remote
def train_remote(seed, dist_type, taus, client_rs, n_clients, T, E_typ,
                 E_cons,gene_process,mode,use_true_q_init=False,a=0.51, b=100,c=2,T_mode='rounds'):
    return train(seed, dist_type, taus, client_rs, n_clients, T, E_typ,
                 E_cons,gene_process,mode,use_true_q_init=use_true_q_init,a=a, b=b,c=c,
                T_mode=T_mode)


def run_federated_simulation(dist_type, taus, client_rs, n_clients, 
                            T,E_typ, E_cons,gene_process, mode, n_sim,use_true_q_init=False, base_seed=2025,
                            a=0.51, b=100,c=2,T_mode='rounds'):

    futures = [train_remote.remote(base_seed + i,
            dist_type, taus, client_rs, n_clients, T,
                                   E_typ, E_cons, gene_process, mode,use_true_q_init=use_true_q_init,
                                   a=a, b=b, c=c, T_mode=T_mode) for i in range(n_sim)]
    results = ray.get(futures)
    return package_results(results)


@ray.remote
def train_history_remote(seed, dist_type, tau, client_rs, n_clients, T, E_typ,
                 E_cons, gene_process, mode, use_true_q_init=False, 
                 a=0.51, b=100, c=2, return_history=True,T_mode='rounds'):
    return train(seed, dist_type, tau, client_rs, n_clients, T, E_typ,
                 E_cons, gene_process, mode, use_true_q_init=use_true_q_init,
                 a=a, b=b, c=c, return_history=return_history, T_mode=T_mode)

def run_federated_trajectory(dist_type, tau, client_rs, n_clients, 
                            T, E_typ, E_cons, gene_process, mode, use_true_q_init=False, base_seed=2025,
                            a=0.51, b=100, c=2,T_mode='rounds'):
    """运行单次联邦训练，固定 return_history=True，返回训练轨迹"""
    
    future = train_history_remote.remote(base_seed, dist_type, tau, client_rs, n_clients, T, E_typ,
                E_cons, gene_process, mode, use_true_q_init=use_true_q_init,
                a=a, b=b, c=c, return_history=True, T_mode=T_mode)
    result = ray.get(future)
    return result