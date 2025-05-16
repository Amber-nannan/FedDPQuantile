import numpy as np
from scipy.stats import norm, cauchy, laplace   # 新增 laplace
import ray

"""
修改函数generate_data，增加参数mu
"""

def generate_data(dist_type, tau, n_samples, mu=0):
    """生成指定均值为mu的分布数据流并返回(数据, 真实分位数)"""
    if dist_type == 'normal':
        data = np.random.normal(mu, 1, n_samples)  # 均值mu，标准差1的正态分布
        true_q = mu + norm.ppf(tau)               # 分位数平移mu
    elif dist_type == 'uniform':
        low, high = mu - 1, mu + 1                # 区间调整为[mu-1, mu+1]，均值为mu
        data = np.random.uniform(low, high, n_samples)
        true_q = low + (high - low) * tau         # 均匀分布分位数公式
    elif dist_type == 'cauchy':
        data = np.random.standard_cauchy(size=n_samples) * 1 + mu  # 位置参数mu的柯西分布
        true_q = mu + cauchy.ppf(tau)             # 分位数平移mu
    elif dist_type == 'laplace':
        data = np.random.laplace(loc=mu, scale=1, size=n_samples)  # Laplace(mu, 1)
        true_q = mu + laplace.ppf(tau)                              # 分位数平移 mu
    
    else:
        raise ValueError("不支持的分布类型。请选择 'normal'、'uniform'、'cauchy' 或 'laplace'")
    
    return data, true_q

def distribute_data(data, n_clients):
    """随机打乱数据并按比例分配到客户端"""
    data = np.random.permutation(data)
    return np.split(data, n_clients)

def package_results(raw_results):
    """统一结果打包格式"""
    true_q, estimates, variances, maes = zip(*raw_results)
    return {
        'estimates': np.array(estimates),
        'variances': np.array(variances),
        'maes': np.array(maes),  # 取最终MAE
        'true_q': np.array(true_q)
    }

def analyze_results(results, z_score=6.74735):
    """分析模拟结果"""
    est = results['estimates']
    var = results['variances']
    true_q = results['true_q']
    
    # 计算覆盖概率
    lower = est - z_score * np.sqrt(var)
    upper = est + z_score * np.sqrt(var)
    coverage = np.mean((true_q >= lower) & (true_q <= upper))
    
    return {
        'coverage': coverage,
        'mae': np.mean(np.abs(est-true_q))
    }

