"""
调参目标：寻找在多种tau、E、T组合下都表现良好的(a,c)值
结合【贝叶斯优化搜策略】/【HyperOpt搜索算法】和【早停策略】
pip install bayesian-optimization==1.4.3
pip install hyperopt
"""

from util_fdp import *
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
import pandas as pd
import numpy as np

E_choices = [1, 5, 'log']
tau_choices = [0.3, 0.5, 0.8]
T_choices = [1000,5000,10000]
# T_choices = [1000,2000,5000,10000,20000,50000]
# 20000,50000
# 把 run_federated_simulation 包装成一个 trainable
def fed_trainable(config):
    all_results = []
    
    # 遍历所有tau, E, T组合
    for tau in tau_choices:
        for E in E_choices:
            for T in T_choices:
                # 根据 E 决定 E_typ / E_cons
                if E == 'log':
                    E_typ, E_cons = 'log', None
                else:
                    E_typ, E_cons = 'cons', int(E)
                
                # 运行模拟
                results = run_federated_simulation(
                    dist_type=config['dist_type'],
                    tau=tau,
                    client_rs=[config["r"]]*config["n_clients"],
                    n_clients=config["n_clients"],
                    T=T,
                    E_typ=E_typ,
                    E_cons=E_cons,
                    gene_process=config["gene_process"],
                    mode=config["mode"],
                    n_sim=config["n_sim"],
                    base_seed=config["base_seed"],
                    a=config["a"], b=config["b"], c=config["c"]
                )
                
                # 分析结果
                z_score = 6.753 if E_typ == 'log' else 6.74735
                output = analyze_results(results, z_score=z_score)
                cvg = output['coverage']
                mae = output['mae']
                
                # 存储结果
                all_results.append({
                    'tau': tau,
                    'E': E,
                    'T': T,
                    'cvg': cvg,
                    'mae': mae
                })
    
    # 计算在所有(tau, E, T)组合下的平均表现
    df_results = pd.DataFrame(all_results)
    avg_cvg = df_results['cvg'].mean()
    avg_mae = df_results['mae'].mean()
    
    # 计算在所有组合下表现良好的比例
    good_ratio = np.mean((df_results['cvg'] > 0.95) & (df_results['mae'] < 0.005))
    
    # 计算综合指标
    # 1. 覆盖率惩罚项：当平均cvg<0.95时，增加惩罚
    cvg_penalty = max(0, 0.95 - avg_cvg) * 10
    
    # 2. MAE惩罚项：当平均mae>0.005时，增加惩罚
    mae_penalty = max(0, avg_mae - 0.005) * 100
    
    # 3. 综合指标：越小越好
    combined_metric = avg_mae + cvg_penalty + mae_penalty
    
    # 4. 额外奖励：根据在所有组合下表现良好的比例
    combined_metric -= good_ratio * 0.1
    
    # 报告结果
    tune.report({
        'avg_cvg': avg_cvg, 
        'avg_mae': avg_mae, 
        'good_ratio': good_ratio,
        'combined_metric': combined_metric,
        'all_results': all_results
    })

if __name__ == "__main__":
    ray.init(runtime_env={"working_dir": "."}) 
    
    # 定义搜索空间（保持不变）
    config = {
        # 超参空间：使用连续空间
        "a": tune.uniform(0.500001, 0.999999),
        "b": 0,
        # "c": tune.uniform(1.0, 5.0),
        # "c": tune.qrandint(1.0, 6.0, 0.5),
        "c": tune.randint(1, 6),
        "dist_type": "normal",  
        "gene_process": "hete",
        "mode": "federated",
        "r": 0.25,    # r=0.25 固定
        # 常数参数
        "n_clients": 10,
        "n_sim": 100,
        "base_seed": 2025
    }

    # # 创建贝叶斯优化搜索器
    # bayesopt = BayesOptSearch(
    #     metric="combined_metric",
    #     mode="min",
    # )
    # 创建HyperOpt搜索器
    hyperopt = HyperOptSearch(
        metric="combined_metric",
        mode="min",
    )
    # 限制并发数量
    search_alg = ConcurrencyLimiter(hyperopt, max_concurrent=12)
    
    # 创建ASHA调度器
    scheduler = ASHAScheduler(
        metric="combined_metric",
        mode="min",
        max_t=1,
        grace_period=1,
        reduction_factor=2
    )

    pg = tune.PlacementGroupFactory(
        [{"CPU": 2}] + [{"CPU": 1}] * config["n_clients"]
    )

    # 调用
    analysis = tune.run(
        fed_trainable,
        config=config,
        metric="combined_metric",
        mode="min",
        search_alg=search_alg,     # 使用贝叶斯优化
        scheduler=scheduler,       # 使用ASHA调度器
        num_samples=50,            # 总共尝试的样本数
        resources_per_trial=pg, 
        max_concurrent_trials=None,
        storage_path="/mnt/ray_tuning/ray_results",
        name="fed_lr_tuning_avg"
    )
    
    # 读取已有实验结果
    restored_analysis = tune.ExperimentAnalysis(
        experiment_checkpoint_path="/mnt/ray_tuning/ray_results/fed_lr_tuning_avg"
    )
    
    # 获取所有试验结果
    df = restored_analysis.results_df
    
    # 筛选满足条件的结果
    good_results = df[
        (df['avg_cvg'] > 0.95) & 
        (df['avg_mae'] < 0.005) &
        (df['good_ratio'] > 0.8)  # 至少80%的组合表现良好
    ]
    
    print("\n最佳 (a,c) 组合：")
    print(good_results[['config/a', 'config/c', 'avg_cvg', 'avg_mae', 'good_ratio']])
    
    # 保存结果到 CSV
    df.to_csv("/mnt/ray_tuning/ray_results/results_avg.csv", index=False)
    
    # 打印最佳配置
    print("Best config: ", analysis.get_best_config(metric="combined_metric", mode="min"))
    print("Best performance: ", analysis.best_result)
    
    ray.shutdown()