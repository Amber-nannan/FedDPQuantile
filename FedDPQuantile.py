import numpy as np
from DPQuantile import DPQuantile

class FedDPQuantile(DPQuantile):
    """联邦差分隐私分位数估计"""
    
    def __init__(self, n_clients=1, pk=None, client_rs=None, a=0.51, b=100,c=2,seed=2025,**kwargs):
        super().__init__(**kwargs)
        self.n_clients = n_clients
        self.pk_init = pk
        self.a = a
        self.b = b
        self.c0 = c
        self.seed = seed
        self.global_q_avg_history = {}    # 添加全局统计量历史记录
        self.global_variance_history = {}
        
        # 处理客户端隐私参数
        self.client_rs = client_rs if client_rs else [self.r] * n_clients
        if len(self.client_rs) != n_clients:
            raise ValueError("client_rs长度必须与n_clients一致")
        self.clients = [self._create_client(i) for i in range(self.n_clients)]
        
        np.random.seed(self.seed)
        q_est = np.random.normal(0, 1)  # 随机初始值
        for clients in self.clients:
            clients.reset(q_est)    # 每个机器相同的随机初始值
            # clients.reset()
        
        # 添加全局统计量历史记录
        self.global_q_avg_history = {}
        self.global_variance_history = {}
            
    def _lr_schedule(self,step,c0=2,a=0.51,b=100):
        """
        学习率策略
        """
        lr = c0 / (step**a + b)  # lr = c0 / (step**a + 500)
        return lr

    def _create_client(self, client_idx):
        """创建客户端实例（带独立隐私参数）"""
        return DPQuantile(tau=self.tau, r=self.client_rs[client_idx],
                          true_q=self.true_q,
                         use_true_q_init=self.use_true_q_init)

    def _get_batch(self, client_idx,Em=1):
        """获取客户端数据批次"""
        try:
            return [next(self.data_streams[client_idx]) for _ in range(Em)]
        except StopIteration:
            return None

    def _aggregate(self, params):
        """参数聚合（默认FedAvg）"""
        return np.average(params, weights=self.pk)

    def fit(self,clients_data,Em_list,warm_up=0.05):
        self.Em_list = Em_list
        self.reset()
        self.data_streams = [iter(data) for data in clients_data]
        self.pk = self.pk_init if self.pk_init else np.ones(self.n_clients)/self.n_clients

        """联邦训练主循环"""
        M = len(self.Em_list)
        warm_up_em = int(np.sum(Em_list) * warm_up)
        for m in range(M):
            m_prime = m - warm_up_em if m > warm_up_em else 0
            Em = self.Em_list[m]
            # 并行本地更新
            client_params = []
            for c in range(self.n_clients):
                batch = self._get_batch(c,Em)
                if batch is None:  # 数据耗尽
                    return 0
                
                client = self.clients[c]
                for x in batch:
                    delta = client._compute_gradient(x)
                    lr = self._lr_schedule(m_prime + 1,c0=self.c0,a=self.a,b=self.b) 
                    client._update_estimator(delta, lr/Em)
                    client._update_stats()
                
                client_params.append(client.q_est)

            # 全局聚合与更新
            global_est = self._aggregate(client_params)
            self._sync_global_state(global_est)

    def _sync_global_state(self, global_est):
        """同步全局状态"""
        # 更新服务器状态
        self.n += 1
        prev_weight = (self.n - 1) / self.n
        self.Q_avg = prev_weight * self.Q_avg + global_est / self.n
        
        # 更新方差统计量
        Em = self.Em_list[self.n-1]
        term = self.n**2 / Em
        self.v_a += term * self.Q_avg**2
        self.v_b += term * self.Q_avg
        self.v_s += 1 / Em
        self.v_q += term
        
        # 记录全局统计量（使用clients[0].n作为索引）
        local_n = self.clients[0].n
        self.global_q_avg_history[local_n] = self.Q_avg
        self.global_variance_history[local_n] = self.get_variance()
        
        # 同步到所有客户端
        for client in self.clients:
            client.q_est = global_est
            client.Q_avg = self.Q_avg
    
    def get_stats_history(self):
        """获取统计量历史记录"""
        # 返回本地客户端的Q_avg历史
        local_stats = {f"client_{i}": client.q_avg_history for i, client in enumerate(self.clients)}
        
        # 返回全局统计量历史
        global_stats = {
            "global_q_avg": self.global_q_avg_history,
            "global_variance": self.global_variance_history
        }
        
        return {
            "local": local_stats,
            "global": global_stats
        }