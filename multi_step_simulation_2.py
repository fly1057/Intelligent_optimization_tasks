import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

import matplotlib.pyplot as plt
import matplotlib

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
        
        # 记录历史数据
        self.P_history = []
        self.SOC_history = []
        self.E_available_history = []
        
    def update_SOC(self, P_new, dt=1):
        # 根据功率和效率更新SOC
        delta_E = P_new * dt / 3600  # kWh
        if P_new > 0:  # 放电
            self.SOC_current -= delta_E / self.E_rated * 100 * (1 / self.eff_discharge)
        else:  # 充电
            self.SOC_current += delta_E / self.E_rated * 100 * self.eff_charge
        self.SOC_current = np.clip(self.SOC_current, self.SOC_min, self.SOC_max)
        self.P_current = P_new
        
        # 记录历史
        self.P_history.append(P_new)
        self.SOC_history.append(self.SOC_current)
        self.E_available_history.append(self.available_energy())
        
    def energy_margin(self, is_charging=False):
        # 计算能量裕度
        if is_charging:
            return (self.SOC_max - self.SOC_current) / 100 * self.E_rated
        else:
            return (self.SOC_current - self.SOC_min) / 100 * self.E_rated
    
    def available_energy(self):
        # 可用能量 = 当前SOC对应的能量
        return self.SOC_current / 100 * self.E_rated
    
    def available_power(self, is_charging=False):
        # 可用功率（考虑当前功率限制和SOC约束）
        if is_charging:
            # 充电：最大可用功率 = min(P_max, 充电功率限制)
            # 还要考虑SOC不能超过上限
            power_from_SOC = (self.SOC_max - self.SOC_current) / 100 * self.E_rated * 3600  # kW·s
            return min(self.P_max, power_from_SOC)
        else:
            # 放电：最大可用功率 = min(P_max, 放电功率限制)
            # 还要考虑SOC不能低于下限
            power_from_SOC = (self.SOC_current - self.SOC_min) / 100 * self.E_rated * 3600  # kW·s
            return min(self.P_max, power_from_SOC)
    
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
# PSO求解器（多步长优化）
# ============================
class MultiStepPSOSolver:
    def __init__(self, systems, dt=1, num_particles=30, max_iter=100):
        self.systems = systems
        self.N = len(systems)
        self.dt = dt
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        # PSO参数
        self.w = 0.8  # 惯性权重
        self.c1 = 1.5  # 个体学习因子
        self.c2 = 1.5  # 群体学习因子
        
    def optimize_step(self, P_cmd):
        """优化单步功率分配"""
        # 初始化粒子群
        positions = []
        velocities = []
        pbest_positions = []
        pbest_values = []
        gbest_position = None
        gbest_value = float('inf')
        
        # 初始化粒子
        for _ in range(self.num_particles):
            P_alloc = self._generate_random_solution(P_cmd)
            positions.append(P_alloc)
            velocities.append(np.random.uniform(-1, 1, self.N))
            pbest_positions.append(P_alloc.copy())
            val = self._fitness(P_alloc, P_cmd)
            pbest_values.append(val)
            if val < gbest_value:
                gbest_value = val
                gbest_position = P_alloc.copy()
        
        # PSO迭代
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # 更新位置
                positions[i] += velocities[i]
                
                # 计算适应度
                fitness = self._fitness(positions[i], P_cmd)
                
                # 更新个体最优
                if fitness < pbest_values[i]:
                    pbest_values[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                
                # 更新全局最优
                if fitness < gbest_value:
                    gbest_value = fitness
                    gbest_position = positions[i].copy()
            
            # 惯性权重衰减
            self.w *= 0.99
        
        return gbest_position, gbest_value
    
    def _generate_random_solution(self, P_cmd):
        # 生成满足总功率平衡的随机解
        P_alloc = np.zeros(self.N)
        
        # 计算各系统的能量裕度比例
        is_charging = P_cmd < 0
        E_margins = [sys.energy_margin(is_charging) for sys in self.systems]
        total_margin = sum(E_margins)
        
        if total_margin > 0:
            # 按能量裕度比例分配
            ratios = [margin / total_margin for margin in E_margins]
            for i in range(self.N):
                P_alloc[i] = ratios[i] * P_cmd
        else:
            # 如果能量裕度为0，按额定容量比例分配
            E_ratios = [sys.E_rated / sum(sys.E_rated for sys in self.systems) for sys in self.systems]
            for i in range(self.N):
                P_alloc[i] = E_ratios[i] * P_cmd
        
        # 调整以满足功率限值
        self._adjust_for_constraints(P_alloc, P_cmd)
        
        return P_alloc
    
    def _adjust_for_constraints(self, P_alloc, P_cmd):
        # 调整功率分配以满足约束
        is_charging = P_cmd < 0
        
        # 确保不超过功率限值
        for i in range(self.N):
            sys = self.systems[i]
            if is_charging:
                P_alloc[i] = np.clip(P_alloc[i], sys.P_min, 0)
            else:
                P_alloc[i] = np.clip(P_alloc[i], 0, sys.P_max)
        
        # 调整总功率平衡
        total_power = sum(P_alloc)
        power_diff = P_cmd - total_power
        
        if abs(power_diff) > 0.01:
            # 重新分配差值
            # 计算各系统的可调整空间
            adjustable_capacity = []
            for i in range(self.N):
                sys = self.systems[i]
                if power_diff > 0:  # 需要增加放电功率
                    available = min(sys.P_max - P_alloc[i], sys.available_power(is_charging=False) - P_alloc[i])
                else:  # 需要减少放电功率（或增加充电功率）
                    available = min(P_alloc[i] - sys.P_min, P_alloc[i] - (-sys.available_power(is_charging=True)))
                adjustable_capacity.append(max(0, available))
            
            total_adjustable = sum(adjustable_capacity)
            if total_adjustable > 0:
                # 按可调整空间比例分配差值
                for i in range(self.N):
                    if adjustable_capacity[i] > 0:
                        adjustment = power_diff * (adjustable_capacity[i] / total_adjustable)
                        P_alloc[i] += adjustment
    
    def _fitness(self, P_alloc, P_cmd):
        # 目标函数：考虑能量裕度分配和SOC趋同性
        is_charging = P_cmd < 0
        
        # 1. 计算总能量裕度
        E_margin_total = sum(sys.energy_margin(is_charging) for sys in self.systems)
        
        # 2. 计算理想分配值（根据能量裕度比例）
        if E_margin_total > 0:
            E_margins = [sys.energy_margin(is_charging) for sys in self.systems]
            P_ideal = [P_cmd * (E_margin / E_margin_total) for E_margin in E_margins]
        else:
            P_ideal = [0] * self.N
        
        # 3. 计算偏差平方和
        fitness_val = sum((P_alloc[i] - P_ideal[i]) ** 2 for i in range(self.N))
        
        # 4. 加入惩罚项（约束违反）
        penalty = 0
        # 总功率平衡约束
        total_power = sum(P_alloc)
        penalty += 1000 * abs(total_power - P_cmd) ** 2
        
        # 个体约束检查
        for i, sys in enumerate(self.systems):
            if not sys.check_constraints(P_alloc[i], self.dt):
                penalty += 1000
        
        return fitness_val + penalty


# ============================
# 多步长仿真主程序
# ============================
class MultiStepSimulation:
    def __init__(self, systems, P_cmd_series, dt=1):
        self.systems = systems
        self.P_cmd_series = P_cmd_series  # 功率指令序列
        self.dt = dt  # 时间步长（秒）
        self.time_steps = len(P_cmd_series)
        
        # 初始化记录数据
        self.records = {
            'time_step': [],
            'timestamp': [],
            'P_cmd': [],
            'total_P_actual': [],
            'total_available_energy': [],
            'total_available_power': []
        }
        
        for sys in systems:
            self.records[f'P_{sys.name}'] = []
            self.records[f'SOC_{sys.name}'] = []
            self.records[f'E_available_{sys.name}'] = []
        
        # 初始化时间
        self.start_time = datetime.now()
        
    def run_simulation(self):
        """运行多步长仿真"""
        print("开始多步长仿真...")
        print(f"总步数: {self.time_steps}")
        print("="*60)
        
        # 创建PSO求解器
        pso_solver = MultiStepPSOSolver(self.systems, dt=self.dt, num_particles=30, max_iter=50)
        
        for step in range(self.time_steps):
            # 当前功率指令
            P_cmd = self.P_cmd_series[step]
            
            # 计算当前总可用能量和功率
            total_available_energy = sum(sys.available_energy() for sys in self.systems)
            is_charging = P_cmd < 0
            total_available_power = sum(sys.available_power(is_charging) for sys in self.systems)
            
            # 如果总可用能量接近耗尽，停止仿真
            if total_available_energy < 0.1:
                print(f"步 {step}: 总可用能量耗尽，停止仿真")
                break
            
            # 如果总可用功率不足以满足指令，调整指令
            actual_P_cmd = P_cmd
            if (P_cmd > 0 and P_cmd > total_available_power) or (P_cmd < 0 and abs(P_cmd) > total_available_power):
                print(f"步 {step}: 可用功率不足，调整指令")
                if P_cmd > 0:
                    actual_P_cmd = total_available_power * 0.95  # 保留5%裕度
                else:
                    actual_P_cmd = -total_available_power * 0.95
            
            # 优化功率分配
            P_alloc, fitness = pso_solver.optimize_step(actual_P_cmd)
            
            # 更新系统状态
            total_P_actual = 0
            for i, sys in enumerate(self.systems):
                sys.update_SOC(P_alloc[i], self.dt)
                total_P_actual += P_alloc[i]
            
            # 记录数据
            current_time = self.start_time + timedelta(seconds=step * self.dt)
            self.records['time_step'].append(step)
            self.records['timestamp'].append(current_time)
            self.records['P_cmd'].append(P_cmd)
            self.records['total_P_actual'].append(total_P_actual)
            self.records['total_available_energy'].append(total_available_energy)
            self.records['total_available_power'].append(total_available_power)
            
            for sys in self.systems:
                self.records[f'P_{sys.name}'].append(sys.P_current)
                self.records[f'SOC_{sys.name}'].append(sys.SOC_current)
                self.records[f'E_available_{sys.name}'].append(sys.available_energy())
            
            # 打印进度
            if step % 10 == 0 or step == self.time_steps - 1:
                print(f"步 {step}: 指令={P_cmd:.1f}kW, 实际={total_P_actual:.1f}kW, "
                      f"可用能量={total_available_energy:.1f}kWh, "
                      f"SOC范围=[{min(sys.SOC_current for sys in self.systems):.1f}%, "
                      f"{max(sys.SOC_current for sys in self.systems):.1f}%]")
        
        print("="*60)
        print("仿真完成!")
        
        return self.records
    
    def save_to_csv(self, filename="储能仿真结果.csv"):
        """保存结果到CSV文件"""
        # 创建DataFrame
        df = pd.DataFrame(self.records)
        
        # 保存到CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {filename}")
        
        return df
    
    def plot_results(self, save_fig=True):
        """绘制结果图表"""
        # 创建输出目录
        if save_fig:
            output_dir = "simulation_results"
            os.makedirs(output_dir, exist_ok=True)
        
        # 时间轴（步数）
        time_steps = self.records['time_step']
        
        # 创建图形
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. 各储能系统功率变化
        ax = axes[0, 0]
        for sys in self.systems:
            ax.plot(time_steps, self.records[f'P_{sys.name}'], label=sys.name, linewidth=2)
        ax.set_xlabel('时间步')
        ax.set_ylabel('功率 (kW)')
        ax.set_title('各储能系统功率变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 各储能系统SOC变化
        ax = axes[0, 1]
        for sys in self.systems:
            ax.plot(time_steps, self.records[f'SOC_{sys.name}'], label=sys.name, linewidth=2)
        ax.set_xlabel('时间步')
        ax.set_ylabel('SOC (%)')
        ax.set_title('各储能系统SOC变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 各储能系统可用能量变化
        ax = axes[1, 0]
        for sys in self.systems:
            ax.plot(time_steps, self.records[f'E_available_{sys.name}'], label=sys.name, linewidth=2)
        ax.set_xlabel('时间步')
        ax.set_ylabel('可用能量 (kWh)')
        ax.set_title('各储能系统可用能量变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 总功率与指令对比
        ax = axes[1, 1]
        ax.plot(time_steps, self.records['P_cmd'], 'k--', label='指令功率', linewidth=2, alpha=0.7)
        ax.plot(time_steps, self.records['total_P_actual'], 'r-', label='实际功率', linewidth=2)
        ax.set_xlabel('时间步')
        ax.set_ylabel('功率 (kW)')
        ax.set_title('总功率与指令对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 总可用能量和可用功率
        ax = axes[2, 0]
        color1 = 'tab:blue'
        ax.plot(time_steps, self.records['total_available_energy'], color=color1, linewidth=2)
        ax.set_xlabel('时间步')
        ax.set_ylabel('总可用能量 (kWh)', color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.set_title('总可用能量变化')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        color2 = 'tab:red'
        ax2.plot(time_steps, self.records['total_available_power'], color=color2, linewidth=2, linestyle='--')
        ax2.set_ylabel('总可用功率 (kW)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 6. 功率分配比例（最后一步）
        ax = axes[2, 1]
        last_step = len(time_steps) - 1
        if last_step >= 0:
            P_values = [self.records[f'P_{sys.name}'][last_step] for sys in self.systems]
            labels = [sys.name for sys in self.systems]
            
            # 只显示非零值
            nonzero_indices = [i for i, val in enumerate(P_values) if abs(val) > 0.01]
            if nonzero_indices:
                nonzero_P = [P_values[i] for i in nonzero_indices]
                nonzero_labels = [labels[i] for i in nonzero_indices]
                
                # 根据充放电选择颜色
                colors = []
                for val in nonzero_P:
                    if val > 0:
                        colors.append('#ff6b6b')  # 红色表示放电
                    else:
                        colors.append('#4ecdc4')  # 青色表示充电
                
                wedges, texts, autotexts = ax.pie(nonzero_P, labels=nonzero_labels, autopct='%1.1f%%', 
                                                  colors=colors, startangle=90)
                
                # 设置文本属性
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.set_title(f'第{last_step+1}步功率分配比例')
            else:
                ax.text(0.5, 0.5, '无功率分配', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('功率分配比例')
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = os.path.join(output_dir, "储能仿真结果.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {fig_path}")
        
        plt.show()
        
        # 创建详细SOC变化图
        plt.figure(figsize=(12, 8))
        for sys in self.systems:
            plt.plot(time_steps, self.records[f'SOC_{sys.name}'], label=sys.name, linewidth=2.5)
        
        # 添加SOC限值线
        for sys in self.systems:
            plt.axhline(y=sys.SOC_min, color='r', linestyle=':', alpha=0.5, linewidth=1)
            plt.axhline(y=sys.SOC_max, color='g', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('SOC (%)', fontsize=12)
        plt.title('储能系统SOC变化趋势（带限值线）', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            fig_path = os.path.join(output_dir, "SOC变化趋势.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        plt.show()


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
    
    # 创建功率指令序列（模拟不同时间的功率需求）
    # 假设前50步放电，中间20步充电，最后30步放电
    total_steps = 3000
    P_cmd_series = []
    
    for step in range(total_steps):
        # if step < 50:
        #     # 前50步：放电，功率逐渐增加
        #     P_cmd = 50 + 30 * np.sin(step / 10) + random.uniform(-5, 5)
        # elif step < 70:
        #     # 中间20步：充电
        #     P_cmd = -30 + 10 * np.sin(step / 5) + random.uniform(-2, 2)
        # else:
        #     # 最后30步：放电，功率逐渐减小
        #     P_cmd = 40 - 0.5 * (step - 70) + 10 * np.sin(step / 8) + random.uniform(-3, 3)

        P_cmd =50
        
        P_cmd_series.append(P_cmd)
    
    # 创建多步长仿真器
    simulation = MultiStepSimulation(systems, P_cmd_series, dt=5)
    
    # 运行仿真
    records = simulation.run_simulation()
    
    # 保存结果到CSV
    df = simulation.save_to_csv("多类型储能系统仿真结果.csv")
    
    # 打印统计信息
    print("\n仿真统计信息:")
    print("="*60)
    print(f"总仿真步数: {len(records['time_step'])}")
    print(f"最终总可用能量: {records['total_available_energy'][-1]:.2f} kWh")
    print(f"平均总功率: {np.mean(records['total_P_actual']):.2f} kW")
    
    print("\n各储能系统最终状态:")
    for sys in systems:
        print(f"{sys.name}: SOC={sys.SOC_current:.2f}%, "
              f"功率={sys.P_current:.2f}kW, "
              f"可用能量={sys.available_energy():.2f}kWh")
    
    # 绘制图表
    simulation.plot_results(save_fig=True)
    
    # 可选：输出部分数据到控制台
    print("\n前10步数据预览:")
    preview_cols = ['time_step', 'P_cmd', 'total_P_actual', 'total_available_energy']
    for sys in systems[:2]:  # 只显示前两个系统
        preview_cols.extend([f'P_{sys.name}', f'SOC_{sys.name}'])
    
    preview_df = df[preview_cols].head(10)
    print(preview_df.to_string())