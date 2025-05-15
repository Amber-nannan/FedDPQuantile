import numpy as np
from typing import Optional



class DPQuantile:
    """差分隐私分位数估计基类"""
        
    def __init__(self, tau=0.5, r=0.5, true_q=None,
     track_history=False, burn_in_ratio=0, use_true_q_init=False,a=0.51,b=100,c=2,
                seed=2025):
        self.tau = tau
        self.r = r
        self.true_q = true_q
        self.track_history = track_history
        self.burn_in_ratio = burn_in_ratio
        self.use_true_q_init = use_true_q_init  # 新增参数
        self.q_avg_history = {}  # 记录每次更新的Q_avg
        self.variance_history = {}  # 记录每次更新的var
        self.a = a
        self.b = b
        self.c0 = c
        self.seed=seed

    def _lr_schedule(self,step,c0=2,a=0.51,b=100):
        """
        学习率策略
        """
        lr = c0 / (step**a + b)  # lr = c0 / (step**a + 500)
        return lr

    def reset(self, q_est: Optional[float]=None):
        """重置训练状态"""
        if self.use_true_q_init and self.true_q is not None:
            self.q_est = self.true_q  # 从真值开始
        elif q_est:
            self.q_est = q_est
        else:
            np.random.seed(self.seed)
            self.q_est = np.random.normal(0,1)
        self.Q_avg = 0.0
        self.n = 0
        self.step = 0
        
        # 在线推断统计量
        self.v_a = 0.0
        self.v_b = 0.0
        self.v_s = 0.0
        self.v_q = 0.0
        self.errors = []

    def _compute_gradient(self, x):
        """核心梯度计算"""
        if np.random.rand() < self.r:
            s = int(x > self.q_est)
        else:
            s = np.random.binomial(1, 0.5)
        
        delta = ((1 - self.r + 2*self.tau*self.r)/2 if s 
                else -(1 + self.r - 2*self.tau*self.r)/2)
        return delta

    def _update_estimator(self, delta, lr):
        """参数更新"""
        self.q_est += lr * delta
        self.step += 1

    def _update_stats(self):
        """更新统计量"""
        self.n += 1
        prev_weight = (self.n - 1) / self.n
        self.Q_avg = prev_weight * self.Q_avg + self.q_est / self.n
        
        
        # 更新方差统计量
        term = self.n**2
        self.v_a += term * self.Q_avg**2
        self.v_b += term * self.Q_avg
        self.v_q += term
        self.v_s += 1

        # 记录当前样本数量对应的Q_avg
        self.q_avg_history[self.n] = self.Q_avg
        self.variance_history[self.n] = self.get_variance()
        
        if self.track_history and self.true_q is not None:
            self.errors.append(np.abs(self.Q_avg - self.true_q))

    def fit(self, data_stream):
        """单机版训练方法"""
        self.reset()
        n_samples = len(data_stream)
        burn_in = int(n_samples * self.burn_in_ratio)  # 计算预热样本数
        for idx, x in enumerate(data_stream):
            # 计算当前步骤的学习率
            lr = self._lr_schedule(self.step + 1,
                            c0=self.c0,a=self.a,b=self.b)
            
            # 计算梯度并更新估计值
            delta = self._compute_gradient(x)
            self._update_estimator(delta, lr)
            
            # 跳过预热阶段的统计量更新
            if idx >= burn_in:
                self._update_stats()
            
            # 提前终止检查
            if self.step >= n_samples:
                break

    def get_stats_history(self):
        stats = {
            "q_avg": self.q_avg_history,
            "variance": self.variance_history
        }
        return stats

    def get_variance(self):
        """获取方差估计"""
        if self.n == 0:
            return 0.0
        return (self.v_a - 2*self.Q_avg*self.v_b + 
               (self.Q_avg**2)*self.v_q) / (self.n**2 * self.v_s)
