import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 或者使用微软雅黑
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False

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
        
        # 历史记录
        self.power_history = []
        self.SOC_history = []
        self.energy_history = []
        self.record_history()
    
    def record_history(self):
        """记录当前状态到历史"""
        self.power_history.append(self.P_current)
        self.SOC_history.append(self.SOC_current)
        self.energy_history.append(self.SOC_current * self.E_rated / 100)  # 当前能量(kWh)
    
    def update_SOC(self, P_new, dt=1):
        """根据功率和效率更新SOC"""
        delta_E = P_new * dt / 3600  # kWh
        if P_new > 0:  # 放电
            self.SOC_current -= delta_E / self.E_rated * 100 * (1 / self.eff_discharge)
        else:  # 充电
            self.SOC_current += delta_E / self.E_rated * 100 * self.eff_charge
        
        # 确保SOC在合理范围内
        self.SOC_current = np.clip(self.SOC_current, self.SOC_min, self.SOC_max)
        self.P_current = P_new
        self.record_history()
    
    def energy_margin(self, is_charging=False):
        """计算能量裕度(kWh)"""
        if is_charging:
            return (self.SOC_max - self.SOC_current) / 100 * self.E_rated
        else:
            return (self.SOC_current - self.SOC_min) / 100 * self.E_rated
    
    def available_power(self, is_charging=False):
        """根据SOC计算可用功率"""
        if is_charging:
            # 充电时，可用功率受限于充电功率限制和SOC上限
            soc_margin = self.SOC_max - self.SOC_current
            power_from_soc = soc_margin / 100 * self.E_rated * 3600 / 1  # kW (dt=1s)
            return min(self.P_max, power_from_soc)
        else:
            # 放电时，可用功率受限于放电功率限制和SOC下限
            soc_margin = self.SOC_current - self.SOC_min
            power_from_soc = soc_margin / 100 * self.E_rated * 3600 / 1 * self.eff_discharge  # kW (dt=1s)
            return min(self.P_max, power_from_soc)
    
    def check_constraints(self, P_new, dt=1):
        """检查功率限制和SOC限制"""
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
    
    def can_continue(self, is_charging=False):
        """检查是否能继续充放电"""
        if is_charging:
            return self.SOC_current < self.SOC_max - 0.1  # 留0.1%的裕量
        else:
            return self.SOC_current > self.SOC_min + 0.1  # 留0.1%的裕量


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
        """初始化粒子群"""
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
        """生成满足总功率平衡的随机解"""
        P_alloc = np.zeros(self.N)
        
        # 计算各系统的可用功率
        is_charging = self.P_cmd < 0
        available_powers = [sys.available_power(is_charging) for sys in self.systems]
        
        # 如果总可用功率无法满足指令，则按比例分配
        total_available = sum(available_powers)
        if is_charging:
            total_available = sum([abs(p) for p in available_powers])
        
        if total_available < abs(self.P_cmd):
            # 按可用功率比例分配
            for i in range(self.N):
                P_alloc[i] = available_powers[i] * self.P_cmd / total_available if total_available > 0 else 0
        else:
            # 首先生成随机比例
            ratios = np.random.dirichlet(np.ones(self.N))
            for i in range(self.N):
                P_alloc[i] = ratios[i] * self.P_cmd
            
            # 调整以满足个体约束
            for i in range(self.N):
                sys = self.systems[i]
                if P_alloc[i] > available_powers[i]:
                    excess = P_alloc[i] - available_powers[i]
                    P_alloc[i] = available_powers[i]
                    # 将多余功率分配给其他系统
                    indices = [j for j in range(self.N) if j != i]
                    for j in indices:
                        if P_alloc[j] < available_powers[j]:
                            addable = available_powers[j] - P_alloc[j]
                            add = min(excess, addable)
                            P_alloc[j] += add
                            excess -= add
                            if excess <= 0:
                                break
                
                elif P_alloc[i] < -available_powers[i] if is_charging else P_alloc[i] < self.systems[i].P_min:
                    if is_charging:
                        shortage = -available_powers[i] - P_alloc[i]
                        P_alloc[i] = -available_powers[i]
                    else:
                        shortage = self.systems[i].P_min - P_alloc[i]
                        P_alloc[i] = self.systems[i].P_min
                    
                    # 从其他系统扣减
                    indices = [j for j in range(self.N) if j != i]
                    for j in indices:
                        if is_charging:
                            if P_alloc[j] > -available_powers[j]:
                                deductable = P_alloc[j] - (-available_powers[j])
                                deduct = min(shortage, deductable)
                                P_alloc[j] -= deduct
                                shortage -= deduct
                        else:
                            if P_alloc[j] > self.systems[j].P_min:
                                deductable = P_alloc[j] - self.systems[j].P_min
                                deduct = min(shortage, deductable)
                                P_alloc[j] -= deduct
                                shortage -= deduct
                        if shortage <= 0:
                            break
        
        return P_alloc
    
    def _fitness(self, P_alloc):
        """计算适应度函数（包含惩罚项）"""
        is_charging = self.P_cmd < 0
        
        # 1. 计算总能量裕度
        E_margin_total = sum(sys.energy_margin(is_charging) for sys in self.systems)
        
        # 2. 计算理想分配值（根据能量裕度比例）
        if E_margin_total > 0:
            if is_charging:
                # 充电时，能量裕度为到SOC_max的距离
                E_margins = [sys.energy_margin(is_charging) for sys in self.systems]
                P_ideal = [self.P_cmd * (E_margin / E_margin_total) for E_margin in E_margins]
            else:
                # 放电时，能量裕度为到SOC_min的距离
                E_margins = [sys.energy_margin(is_charging) for sys in self.systems]
                P_ideal = [self.P_cmd * (E_margin / E_margin_total) for E_margin in E_margins]
        else:
            P_ideal = [0] * self.N
        
        # 3. 计算偏差平方和（目标函数J）
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
        """运行PSO求解"""
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
            
            # 惯性权重衰减
            self.w *= 0.99
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.gbest_value:.4f}")
        
        return self.gbest_position, self.gbest_value


# ============================
# 多步长模拟器
# ============================
class MultiStepSimulator:
    def __init__(self, systems, P_cmd, dt=1, max_steps=1000):
        self.systems = systems
        self.P_cmd = P_cmd
        self.dt = dt
        self.max_steps = max_steps
        self.is_charging = P_cmd < 0
        
        # 模拟结果
        self.time_steps = []
        self.total_power = []
        self.allocations = []  # 每个时间步的功率分配
        
    def simulate(self):
        """执行多步长模拟"""
        step = 0
        
        while step < self.max_steps:
            # 检查是否所有系统都无法继续
            can_continue = any(sys.can_continue(self.is_charging) for sys in self.systems)
            if not can_continue:
                print(f"所有储能系统已{'充满' if self.is_charging else '放空'}，模拟结束")
                break
            
            # 计算当前总可用功率
            total_available = sum(sys.available_power(self.is_charging) for sys in self.systems)
            
            # 如果可用功率不足，调整指令
            effective_cmd = self.P_cmd
            if self.is_charging:
                if total_available < abs(self.P_cmd):
                    effective_cmd = -total_available  # 按最大可用充电功率
            else:
                if total_available < self.P_cmd:
                    effective_cmd = total_available  # 按最大可用放电功率
            
            # 使用PSO求解功率分配
            solver = PSOSolver(self.systems, effective_cmd, self.dt, num_particles=30, max_iter=50)
            best_solution, best_fitness = solver.solve()
            
            # 记录结果
            self.time_steps.append(step)
            self.total_power.append(effective_cmd)
            self.allocations.append(best_solution.copy())
            
            # 更新系统状态
            for i, sys in enumerate(self.systems):
                sys.update_SOC(best_solution[i], self.dt)
            
            step += 1
            
            # 每100步打印一次进度
            if step % 100 == 0:
                avg_soc = np.mean([sys.SOC_current for sys in self.systems])
                print(f"步长 {step}: 平均SOC = {avg_soc:.2f}%")
        
        print(f"模拟完成，共运行 {step} 个时间步")
        return step
    
    def get_results(self):
        """获取模拟结果"""
        # 创建时间序列数据
        time_seconds = np.array(self.time_steps) * self.dt
        time_hours = time_seconds / 3600
        
        # 提取各系统的功率、SOC和能量
        results = {
            'time_seconds': time_seconds,
            'time_hours': time_hours,
            'total_power': self.total_power
        }
        
        for i, sys in enumerate(self.systems):
            results[f'{sys.name}_power'] = [alloc[i] for alloc in self.allocations]
            results[f'{sys.name}_SOC'] = sys.SOC_history[:len(self.time_steps)]
            results[f'{sys.name}_energy'] = sys.energy_history[:len(self.time_steps)]
        
        return results
    
    def plot_results(self):
        """绘制结果图表"""
        results = self.get_results()
        time_hours = results['time_hours']
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. 总功率变化
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_hours, results['total_power'], 'k-', linewidth=2)
        ax1.fill_between(time_hours, 0, results['total_power'], where=np.array(results['total_power']) > 0, 
                         alpha=0.3, color='red', label='放电')
        ax1.fill_between(time_hours, 0, results['total_power'], where=np.array(results['total_power']) < 0, 
                         alpha=0.3, color='blue', label='充电')
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('总功率 (kW)')
        ax1.set_title('储能电站总功率变化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 各系统功率分配
        ax2 = fig.add_subplot(gs[1, 0])
        # 取最后一个时间步的功率分配
        last_step = len(time_hours) - 1
        if last_step >= 0:
            system_names = [sys.name for sys in self.systems]
            powers = [results[f'{name}_power'][last_step] for name in system_names]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            bars = ax2.bar(system_names, powers, color=colors[:len(system_names)])
            ax2.set_xlabel('储能系统')
            ax2.set_ylabel('功率 (kW)')
            ax2.set_title(f'最后时间步功率分配 (t={time_hours[last_step]:.2f}h)')
            ax2.grid(True, alpha=0.3, axis='y')
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
        
        # 3. SOC变化趋势
        ax3 = fig.add_subplot(gs[1, 1:])
        for i, sys in enumerate(self.systems):
            soc_data = results[f'{sys.name}_SOC']
            ax3.plot(time_hours, soc_data, label=sys.name, linewidth=2)
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('SOC (%)')
        ax3.set_title('各储能系统SOC变化趋势')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 各系统能量变化
        ax4 = fig.add_subplot(gs[2, 0])
        energy_data = []
        labels = []
        for sys in self.systems:
            if len(results[f'{sys.name}_energy']) > 0:
                energy_data.append(results[f'{sys.name}_energy'][-1])
                labels.append(sys.name)
        if energy_data:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            wedges, texts, autotexts = ax4.pie(energy_data, labels=labels, autopct='%1.1f%%', 
                                                colors=colors[:len(labels)], startangle=90)
            ax4.set_title('最终能量分布')
        
        # 5. 各系统功率贡献堆叠图
        ax5 = fig.add_subplot(gs[2, 1:])
        # 准备堆叠数据
        stack_data = []
        for sys in self.systems:
            if len(results[f'{sys.name}_power']) > 0:
                # 只取部分数据点以避免图表过于密集
                step = max(1, len(time_hours) // 100)
                sampled_time = time_hours[::step]
                sampled_power = results[f'{sys.name}_power'][::step]
                stack_data.append(sampled_power)
        
        if stack_data:
            ax5.stackplot(sampled_time, stack_data, labels=[sys.name for sys in self.systems], alpha=0.8)
            ax5.set_xlabel('时间 (小时)')
            ax5.set_ylabel('功率 (kW)')
            ax5.set_title('各储能系统功率贡献（堆叠图）')
            ax5.grid(True, alpha=0.3)
            ax5.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # 打印汇总统计
        self.print_statistics(results)
    
    def print_statistics(self, results):
        """打印统计信息"""
        print("\n" + "="*60)
        print("模拟统计信息")
        print("="*60)
        
        total_duration = results['time_seconds'][-1] if len(results['time_seconds']) > 0 else 0
        total_energy = 0
        
        for sys in self.systems:
            if len(results[f'{sys.name}_power']) > 0:
                # 计算总能量吞吐量 (kWh)
                power_array = np.array(results[f'{sys.name}_power'])
                time_array = np.array(results['time_seconds'])
                # 使用梯形法计算积分
                energy_throughput = np.trapz(np.abs(power_array), time_array) / 3600  # kWh
                total_energy += energy_throughput
                
                print(f"\n{sys.name}:")
                print(f"  初始SOC: {sys.SOC_history[0]:.2f}%")
                print(f"  最终SOC: {sys.SOC_history[-1]:.2f}%")
                print(f"  SOC变化: {sys.SOC_history[-1] - sys.SOC_history[0]:.2f}%")
                print(f"  能量吞吐量: {energy_throughput:.2f} kWh")
                print(f"  最大功率: {np.max(np.abs(power_array)):.2f} kW")
                print(f"  平均功率: {np.mean(np.abs(power_array)):.2f} kW")
        
        print(f"\n总计:")
        print(f"  总运行时间: {total_duration/3600:.2f} 小时")
        print(f"  总能量吞吐量: {total_energy:.2f} kWh")
        print(f"  平均总功率: {np.mean(np.abs(results['total_power'])):.2f} kW")
        
        # 计算SOC趋同性指标（标准差）
        final_socs = [sys.SOC_history[-1] for sys in self.systems]
        soc_std = np.std(final_socs)
        print(f"  最终SOC标准差: {soc_std:.2f}% (越小表示趋同性越好)")


# ============================
# 主程序
# ============================
if __name__ == "__main__":
    # 定义四个储能系统（修改为更合理的参数）
    # systems = [
    #     EnergyStorageSystem("氢燃料电池", E_rated=500, P_current=0, P_min=-60, P_max=60, SOC_current=50, SOC_min=5, SOC_max=80, dP_max=1, eff_charge=0.95, eff_discharge=0.95),
    #     EnergyStorageSystem("铁硫电池",   E_rated=20, P_current=0, P_min=-5, P_max=5,   SOC_current=60, SOC_min=5, SOC_max=90, dP_max=3, eff_charge=0.92, eff_discharge=0.92),
    #     EnergyStorageSystem("钠电池",     E_rated=10, P_current=0, P_min=-10, P_max=10, SOC_current=70, SOC_min=5, SOC_max=95, dP_max=10, eff_charge=0.98, eff_discharge=0.98),
    #     EnergyStorageSystem("钒液流电池", E_rated=10, P_current=0, P_min=-5, P_max=5,   SOC_current=55, SOC_min=5, SOC_max=85, dP_max=3,  eff_charge=0.90, eff_discharge=0.90)
    # ]

    systems = [
        EnergyStorageSystem("氢燃料电池", E_rated=200, P_current=0, P_min=-60, P_max=60, SOC_current=50, SOC_min=5, SOC_max=80, dP_max=1, eff_charge=0.95, eff_discharge=0.95),
        EnergyStorageSystem("铁硫电池",   E_rated=100, P_current=0, P_min=-5, P_max=5,   SOC_current=60, SOC_min=5, SOC_max=90, dP_max=3, eff_charge=0.92, eff_discharge=0.92),
        EnergyStorageSystem("钠电池",     E_rated=100, P_current=0, P_min=-30, P_max=30, SOC_current=70, SOC_min=5, SOC_max=95, dP_max=30, eff_charge=0.98, eff_discharge=0.98),
        EnergyStorageSystem("钒液流电池", E_rated=100, P_current=0, P_min=-5, P_max=5,   SOC_current=55, SOC_min=5, SOC_max=85, dP_max=3,  eff_charge=0.90, eff_discharge=0.90)
    ]
    
    # 电力调度功率指令（放电为正，充电为负）
    P_cmd = 50  # kW，放电指令
    
    print("开始多步长模拟...")
    print(f"调度指令: {P_cmd} kW (放电)")
    print(f"储能系统数量: {len(systems)}")
    
    # 创建多步长模拟器
    simulator = MultiStepSimulator(systems, P_cmd, dt=5, max_steps=10000)  # 时间步长5秒
    
    # 执行模拟
    steps = simulator.simulate()
    
    # 绘制结果
    simulator.plot_results()
    
    # 可选：保存结果到CSV文件
    results = simulator.get_results()
    df = pd.DataFrame(results)
    df.to_csv('energy_storage_simulation_results.csv', index=False)
    print("结果已保存到 energy_storage_simulation_results.csv")