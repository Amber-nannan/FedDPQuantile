"""
普通的【网格搜索】，可以用来打印明细结果(如果需要)
"""
from util_fdp import *
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# r = 0.25, 找一组a,c，# r = 0.25, 找一组a,c，对tau = [0.3,0.5,0.8], E = [1,5,'log'], T=1000/2000/10000/20000 表现好（指cvg大且mae小）

E_choices = ['log']
# E_choices = [1,4,5,10,'log']
tau_choices = [0.3, 0.5, 0.8]
T_choices = [20000,50000]
# T_choices = [1000,2000,5000,10000,20000,50000]

# 把 run_federated_simulation 包装成一个 trainable
def fed_trainable(config):
    E = config["E"]
    # 根据 E 决定 E_typ / E_cons
    if E == 'log':
        E_typ, E_cons = 'log', None
    else:
        E_typ, E_cons = 'cons', int(E)

    results = run_federated_simulation(
        dist_type=config["dist_type"],
        tau=config["tau"],
        client_rs=[config["r"]]*config["n_clients"],
        n_clients=config["n_clients"],
        T=config["T"],
        E_typ=E_typ,
        E_cons=E_cons,
        gene_process=config["gene_process"],
        mode=config["mode"],
        n_sim=config["n_sim"],
        base_seed=config["base_seed"],
        a=config["a"], b=config["b"], c=config["c"]
    )

    z_score = 6.753 if E_typ == 'log' else 6.74735
    output = analyze_results(results, z_score=z_score)
    cvg = output['coverage']
    mae = output['mae']
    tune.report({'cvg':cvg,'mae':mae})


if __name__ == "__main__":
    ray.init(runtime_env={"working_dir": "."}) 
    # 定义搜索空间
    config = {
        # 超参空间：用网格搜索
        "a": tune.grid_search([0.51, 0.6, 0.7, 0.8, 0.9, 0.99]),
        "b": tune.grid_search([0,25,50,75,100]),
        "c": tune.grid_search([1]),
        "tau": tune.grid_search(tau_choices),
        "E": tune.grid_search(E_choices),
        "T": tune.grid_search(T_choices),
        "dist_type": "normal",  
        "gene_process": "hete",
        # "gene_process": tune.grid_search([0.5, 0.3, 0.1]),
        "mode": "federated",
        "r": 0.9,    # r=0.25 固定
        # 常数参数
        "n_clients": 10,
        "n_sim": 100,
        "base_seed": 2025,
    }

    pg = tune.PlacementGroupFactory(
        [{"CPU": 2}] + [{"CPU": 1}] * config["n_clients"]
    )

    from datetime import datetime, timedelta
    now = datetime.utcnow() + timedelta(hours=8)  #（东八区）时间
    suffix = now.strftime("%Y%m%d%H%M%S")

    # 调用
    analysis = tune.run(
        fed_trainable,
        config=config,
        scheduler=None,  # 注意：ASHAScheduler 只能盯一个指标，我们这里不传 scheduler 让它跑完所有 trial
        resources_per_trial=pg, 
        max_concurrent_trials=12,  # 限制最大并行试验数
        storage_path="/mnt/ray_tuning/ray_results",         # 结果输出目录
        name=f"fed_lr_tuning_{suffix}"
    )
    
    # 读取已有实验结果
    restored_analysis = tune.ExperimentAnalysis(
        experiment_checkpoint_path=f"/mnt/ray_tuning/ray_results/fed_lr_tuning_{suffix}"
    )
    
    # 获取所有试验结果
    df = restored_analysis.results_df
    
    # 保存结果到 CSV
    df.to_csv(f"/mnt/ray_tuning/ray_results/results_{suffix}.csv", index=False)
    # grouped_results.to_csv("/mnt/ray_tuning/ray_results/grouped_results.csv", index=False)
