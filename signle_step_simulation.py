import numpy as np
import random

# ============================
# 储能系统类定义
# ============================
class EnergyStorageSystem:
    def __init__(self, name, E_rated, P_current, P_min, P_max, SOC_current, SOC_min, SOC_max, dP_max, eff_charge, eff_discharge):
        self.name = name
        self.E_rated = E_rated  # kWh
        self.P_current = P_current  # kW
        self.P_min = P_min  # kW
        self.P_max = P_max  # kW
        self.SOC_current = SOC_current  # %
        self.SOC_min = SOC_min  # %
        self.SOC_max = SOC_max  # %
        self.dP_max = dP_max  # kW/s
        self.eff_charge = eff_charge  # 充电效率
        self.eff_discharge = eff_discharge  # 放电效率

    def update_SOC(self, P_new, dt=1):
        # 根据功率和效率更新SOC
        delta_E = P_new * dt / 3600  # kWh
        if P_new > 0:  # 放电
            self.SOC_current -= delta_E / self.E_rated * 100 * (1 / self.eff_discharge)
        else:  # 充电
            self.SOC_current += delta_E / self.E_rated * 100 * self.eff_charge
        self.SOC_current = np.clip(self.SOC_current, self.SOC_min, self.SOC_max)

    def energy_margin(self, is_charging=False):
        # 计算能量裕度
        if is_charging:
            return (self.SOC_max - self.SOC_current) / 100 * self.E_rated
        else:
            return (self.SOC_current - self.SOC_min) / 100 * self.E_rated

    def check_constraints(self, P_new, dt=1):
        # 检查功率限制和SOC限制
        if P_new < self.P_min or P_new > self.P_max:
            return False
        if abs(P_new - self.P_current) > self.dP_max * dt:
            return False
        # 预测SOC
        temp_SOC = self.SOC_current
        delta_E = P_new * dt / 3600
        if P_new > 0:
            temp_SOC -= delta_E / self.E_rated * 100 * (1 / self.eff_discharge)
        else:
            temp_SOC += delta_E / self.E_rated * 100 * self.eff_charge
        if temp_SOC < self.SOC_min or temp_SOC > self.SOC_max:
            return False
        return True


# ============================
# PSO求解器
# ============================
class PSOSolver:
    def __init__(self, systems, P_cmd, dt=1, num_particles=30, max_iter=100):
        self.systems = systems
        self.N = len(systems)
        self.P_cmd = P_cmd
        self.dt = dt
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        # PSO参数
        self.w = 0.8  # 惯性权重
        self.c1 = 1.5  # 个体学习因子
        self.c2 = 1.5  # 群体学习因子
        
        # 初始化粒子位置和速度
        self.positions = []
        self.velocities = []
        self.pbest_positions = []
        self.pbest_values = []
        self.gbest_position = None
        self.gbest_value = float('inf')
        
        self._init_particles()
    
    def _init_particles(self):
        for _ in range(self.num_particles):
            # 随机生成满足总功率平衡的初始解
            P_alloc = self._generate_random_solution()
            self.positions.append(P_alloc)
            self.velocities.append(np.random.uniform(-1, 1, self.N))
            self.pbest_positions.append(P_alloc.copy())
            val = self._fitness(P_alloc)
            self.pbest_values.append(val)
            if val < self.gbest_value:
                self.gbest_value = val
                self.gbest_position = P_alloc.copy()
    
    def _generate_random_solution(self):
        # 生成满足总功率平衡的随机解
        P_alloc = np.zeros(self.N)
        remaining = self.P_cmd
        
        # 首先生成随机比例
        ratios = np.random.dirichlet(np.ones(self.N))
        for i in range(self.N):
            P_alloc[i] = ratios[i] * self.P_cmd
        
        # 调整以满足个体约束
        for i in range(self.N):
            sys = self.systems[i]
            if P_alloc[i] > sys.P_max:
                excess = P_alloc[i] - sys.P_max
                P_alloc[i] = sys.P_max
                # 将多余功率分配给其他系统
                indices = [j for j in range(self.N) if j != i]
                for j in indices:
                    if P_alloc[j] < self.systems[j].P_max:
                        addable = self.systems[j].P_max - P_alloc[j]
                        add = min(excess, addable)
                        P_alloc[j] += add
                        excess -= add
                        if excess <= 0:
                            break
            
            elif P_alloc[i] < sys.P_min:
                shortage = sys.P_min - P_alloc[i]
                P_alloc[i] = sys.P_min
                # 从其他系统扣减
                indices = [j for j in range(self.N) if j != i]
                for j in indices:
                    if P_alloc[j] > self.systems[j].P_min:
                        deductable = P_alloc[j] - self.systems[j].P_min
                        deduct = min(shortage, deductable)
                        P_alloc[j] -= deduct
                        shortage -= deduct
                        if shortage <= 0:
                            break
        
        return P_alloc
    
    def _fitness(self, P_alloc):
        # 目标函数：考虑能量裕度分配和SOC趋同性
        is_charging = self.P_cmd < 0
        
        # 1. 计算总能量裕度
        E_margin_total = sum(sys.energy_margin(is_charging) for sys in self.systems)
        
        # 2. 计算理想分配值（根据能量裕度比例）
        if is_charging:
            # 充电时，能量裕度为到SOC_max的距离
            E_margins = [sys.energy_margin(is_charging) for sys in self.systems]
            P_ideal = [self.P_cmd * (E_margin / E_margin_total) for E_margin in E_margins]
        else:
            # 放电时，能量裕度为到SOC_min的距离
            E_margins = [sys.energy_margin(is_charging) for sys in self.systems]
            P_ideal = [self.P_cmd * (E_margin / E_margin_total) for E_margin in E_margins]
        
        # 3. 计算偏差平方和
        fitness_val = sum((P_alloc[i] - P_ideal[i]) ** 2 for i in range(self.N))
        
        # 4. 加入惩罚项（约束违反）
        penalty = 0
        # 总功率平衡约束
        total_power = sum(P_alloc)
        penalty += 1000 * abs(total_power - self.P_cmd) ** 2
        
        # 个体约束检查
        for i, sys in enumerate(self.systems):
            if not sys.check_constraints(P_alloc[i], self.dt):
                penalty += 1000
        
        return fitness_val + penalty
    
    def solve(self):
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                # 更新位置
                self.positions[i] += self.velocities[i]
                
                # 计算适应度
                fitness = self._fitness(self.positions[i])
                
                # 更新个体最优
                if fitness < self.pbest_values[i]:
                    self.pbest_values[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                
                # 更新全局最优
                if fitness < self.gbest_value:
                    self.gbest_value = fitness
                    self.gbest_position = self.positions[i].copy()
            
            # 可选的：惯性权重衰减
            self.w *= 0.99
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.gbest_value:.4f}")
        
        return self.gbest_position, self.gbest_value


# ============================
# 主程序
# ============================
if __name__ == "__main__":
    # 定义四个储能系统（示例参数）
    systems = [
        EnergyStorageSystem("氢燃料电池", E_rated=120, P_current=0, P_min=-70, P_max=60, 
                           SOC_current=50, SOC_min=20, SOC_max=80, dP_max=1, 
                           eff_charge=0.95, eff_discharge=0.95),
        EnergyStorageSystem("铁硫电池", E_rated=10, P_current=0, P_min=-5, P_max=5, 
                           SOC_current=60, SOC_min=30, SOC_max=90, dP_max=5, 
                           eff_charge=0.92, eff_discharge=0.92),
        EnergyStorageSystem("钠电池", E_rated=100, P_current=0, P_min=-20, P_max=20, 
                           SOC_current=70, SOC_min=40, SOC_max=95, dP_max=10, 
                           eff_charge=0.98, eff_discharge=0.98),
        EnergyStorageSystem("钒液流电池", E_rated=10, P_current=0, P_min=-3, P_max=3, 
                           SOC_current=55, SOC_min=25, SOC_max=85, dP_max=1, 
                           eff_charge=0.90, eff_discharge=0.90)
    ]
    
    # 电力调度功率指令（负值为充电，正值为放电）
    P_cmd = 120  # kW，放电指令
    
    # 创建PSO求解器
    solver = PSOSolver(systems, P_cmd, dt=1, num_particles=30, max_iter=100)
    
    # 求解
    best_solution, best_fitness = solver.solve()
    
    print("\n" + "="*50)
    print("优化结果：")
    print(f"总指令功率: {P_cmd} kW")
    print(f"最佳适应度: {best_fitness:.4f}")
    print("-"*30)
    
    total = 0
    for i, sys in enumerate(systems):
        print(f"{sys.name}: {best_solution[i]:.2f} kW")
        total += best_solution[i]
    
    print(f"总输出功率: {total:.2f} kW")
    
    # 更新系统状态
    for i, sys in enumerate(systems):
        sys.update_SOC(best_solution[i], dt=1)
        print(f"{sys.name} 更新后SOC: {sys.SOC_current:.2f}%")