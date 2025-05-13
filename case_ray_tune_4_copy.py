"""
调参目标：寻找在多种tau、E、T组合下都表现良好的(a,c)值
结合【贝叶斯优化搜策略】/【HyperOpt搜索算法】和【早停策略】
"""

# pip install bayesian-optimization==1.4.3
# pip install hyperopt
from util_fdp import *
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
import pandas as pd
import numpy as np

# E_choices = [1, 5, 'log']
# r_choices = [0.25, 0.5, 0.9]
tau_choices = [0.3, 0.5, 0.8]
T_choices = [20000,50000]

# 把 run_federated_simulation 包装成一个 trainable
def fed_trainable(config):
    all_results = []
    
    # 遍历所有tau, T组合，但r和E由命令行参数指定
    for tau in tau_choices:
        for T in T_choices:
            # 根据 E 决定 E_typ / E_cons
            if config["E"] == 'log':
                E_typ, E_cons = 'log', None
            else:
                E_typ, E_cons = 'cons', int(config["E"])
            
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
            
            # 存储结果，添加r和E
            all_results.append({
                'tau': tau,
                'r': config["r"],
                'E': config["E"],
                'T': T,
                'cvg': cvg,
                'mae': mae
            })
    
    # 计算在所有(tau, T)组合下的平均表现
    df_results = pd.DataFrame(all_results)
    avg_cvg = df_results['cvg'].mean()
    avg_mae = df_results['mae'].mean()
    
    # 计算在所有组合下表现良好的比例
    good_ratio = np.mean((df_results['cvg'] > 0.90) & (df_results['mae'] < 0.008))
    
    # 计算综合指标
    # 硬约束：如果覆盖率太低或MAE太高，给予极大惩罚
    hard_penalty = 0
    if avg_cvg < 0.8 or avg_mae > 0.04:
        hard_penalty = 100

    # 软约束：在可接受范围内的优化
    cvg_soft = max(0, 0.95 - avg_cvg) * 5
    mae_soft = max(0, avg_mae - 0.008) * 50

    # 综合指标
    combined_metric = hard_penalty + cvg_soft + mae_soft + avg_mae

    # good_ratio奖励
    combined_metric -= good_ratio * 0.2
    
    # 报告结果
    tune.report({
        'avg_cvg': avg_cvg, 
        'avg_mae': avg_mae, 
        'good_ratio': good_ratio,
        'combined_metric': combined_metric,
        'all_results': all_results
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='联邦学习调参实验')
    parser.add_argument('-E', type=str, default='log', help='E值 (可选: 1, 5, log)')
    parser.add_argument('-r', type=float, default=0.5, help='响应率r值 (默认: 0.5)')
    args = parser.parse_args()

    ray.init(runtime_env={"working_dir": "."}) 
    
    # 定义搜索空间
    config = {
        # 超参空间：使用连续空间
        "a": tune.uniform(0.500001, 0.999999),
        "b": tune.choice([0,25,50,75,100,125,150,175,200]),
        "c": 2,
        "dist_type": "normal",  
        "gene_process": "hete",
        "mode": "federated",
        "E": args.E,
        "r": args.r,  # 添加r参数
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
    search_alg = ConcurrencyLimiter(hyperopt, max_concurrent=20)
    
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

    from datetime import datetime, timedelta
    now = datetime.utcnow() + timedelta(hours=8)  #（东八区）时间
    suffix = now.strftime("%Y%m%d%H%M%S")
    
    # 在文件名中添加r和E的值
    r_str = str(args.r).replace('.', '_')
    E_str = str(args.E).replace('.', '_')
    experiment_name = f"fed_lr_tuning_r{r_str}_E{E_str}_{suffix}"

    # 调用
    analysis = tune.run(
        fed_trainable,
        config=config,
        search_alg=search_alg,     # 使用贝叶斯优化
        scheduler=scheduler,       # 使用ASHA调度器
        num_samples=50,            # 总共尝试的样本数
        resources_per_trial=pg, 
        max_concurrent_trials=None,
        storage_path="/mnt/ray_tuning/ray_results",
        name=experiment_name
    )

    
    # 读取已有实验结果
    restored_analysis = tune.ExperimentAnalysis(
        experiment_checkpoint_path=f"/mnt/ray_tuning/ray_results/{experiment_name}"
    )
    
    # 获取所有试验结果
    df = restored_analysis.results_df
    
    # 保存结果到 CSV
    df.to_csv(f"/mnt/ray_tuning/ray_results/results_{experiment_name}.csv", index=False)
    
    ray.shutdown()