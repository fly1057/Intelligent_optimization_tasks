import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd


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
        delta_E_Grid = P_new * dt / 3600  # kWh  电网接口变化的能量

        if P_new > 0:  # 放电
            # 电池放出的能量 = 电网接口变化的能量 / 放电效率
            delta_E_Batt = delta_E_Grid / self.eff_discharge
            self.SOC_current = self.SOC_current - delta_E_Batt / self.E_rated * 100
        elif P_new <0:  # 充电
            # 电池储存的能量 = 电网接口变化的能量 * 充电效率
            delta_E_Batt = delta_E_Grid * self.eff_charge
            self.SOC_current = self.SOC_current + abs(delta_E_Batt) / self.E_rated * 100
        else :#P_new ==0
            pass
        # 确保SOC在合理范围内
        self.SOC_current = np.clip(self.SOC_current, self.SOC_min, self.SOC_max) #限定在SOC_min和SOC_max之间
        self.P_current = P_new

        # 记录历史
        self.record_history()

    def energy_margin(self, is_charging=False):
        """
        计算能量裕度，用于功率分配
        这里返回的是不考虑效率的物理能量裕度
        """
        if True == is_charging:
            # 充电裕度：还能充多少能量（不考虑效率）
            return (self.SOC_max - self.SOC_current) / 100 * self.E_rated
        elif False == is_charging:
            # 放电裕度：还能放多少能量（不考虑效率）
            return (self.SOC_current - self.SOC_min) / 100 * self.E_rated

    def available_energy(self):
        """可用能量 = 当前SOC对应的能量"""
        return self.SOC_current / 100 * self.E_rated

    def available_power(self, is_charging=False, dt=1):
        """
        可用功率，考虑物理约束
        计算现有能量能够支撑的最大充放电功率，是带正负号的
        """
        if is_charging:
            # 充电：基于SOC上限计算最大充电功率
            max_charge_energy = (self.SOC_max - self.SOC_current) / 100 * self.E_rated
            # 转换为功率：能量/时间，考虑dt
            max_power_from_SOC = max_charge_energy * 3600 / dt  # kWh->kW*s，再除以dt
            # 取功率限制和SOC限制的最小值
            return max(self.P_min, min(0, -max_power_from_SOC))  # 充电功率为负

        else:
            # 放电：基于SOC下限计算最大放电功率
            max_discharge_energy = (self.SOC_current - self.SOC_min) / 100 * self.E_rated
            # 转换为功率
            max_power_from_SOC = max_discharge_energy * 3600 / dt
            # 取功率限制和SOC限制的最小值
        return min(self.P_max, max(0, max_power_from_SOC))  # 放电功率为正

    def check_constraints(self, P_new, dt=1):
        """
        检查功率约束,预测SOC变化时不考虑效率
        只检查物理约束是否允许
        """
        # 1. 功率边界检查
        if P_new < self.P_min or P_new > self.P_max:
            return False

        # 2. 功率变化率检查
        if abs(P_new - self.P_current) > self.dP_max * dt:
            return False

        # 3. 预测SOC（不考虑效率，只检查物理可行性）
        delta_E_kWh = P_new * dt / 3600  # 能量变化（不考虑效率）

        if P_new > 0:  # 放电
            # 最大可能放出的能量（考虑SOC下限）
            max_discharge_energy = (self.SOC_current - self.SOC_min) / 100 * self.E_rated

            # 检查是否有足够的能量放电
            if delta_E_kWh > max_discharge_energy:
                return False

            # 预测SOC（不考虑效率）
            predicted_SOC = self.SOC_current - (delta_E_kWh / self.E_rated) * 100

        elif P_new < 0:  # 充电
            # 最大可能充入的能量（考虑SOC上限）
            max_charge_energy = (self.SOC_max - self.SOC_current) / 100 * self.E_rated

            # 检查是否有足够的空间充电
            if abs(delta_E_kWh) > max_charge_energy:
                return False

            # 预测SOC（不考虑效率）
            predicted_SOC = self.SOC_current + (delta_E_kWh / self.E_rated) * 100
            # 注意：因为delta_E_kWh为负，所以是增加SOC

        else:  # P_new = 0
            predicted_SOC = self.SOC_current

        # 检查预测SOC是否在允许范围内
        if predicted_SOC < self.SOC_min or predicted_SOC > self.SOC_max:
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
    def __init__(self, systems, P_cmd, dt=1, num_particles=30, max_iter=100, previous_solution=None):
        self.systems = systems
        self.N = len(systems)
        self.P_cmd = P_cmd
        self.dt = dt
        self.num_particles = num_particles  #粒子数
        self.max_iter = max_iter            #最大迭代次数
        self.previous_solution = previous_solution  # 上一步的最优解

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
        for i in range(self.num_particles):  # 对每个粒子执行以下操作，每个粒子就是一个解
            # 1. 生成满足约束的功率分配方案
            if i == 0 and self.previous_solution is not None:
                # 第一个粒子使用上一步的最优解
                P_alloc = self.previous_solution.copy()
            else:
                # 其他粒子生成随机解
                P_alloc = self._generate_random_solution()

                # 如果有上一步最优解，在其附近生成粒子
                if self.previous_solution is not None:
                    # 添加小扰动，保持在合理范围内
                    perturbation = np.random.uniform(-0.1, 0.1, self.N) * self.P_cmd
                    P_alloc = self.previous_solution + perturbation

                    # 确保总功率平衡
                    total_power = np.sum(P_alloc)
                    correction = (total_power - self.P_cmd) / self.N
                    P_alloc -= correction

                    # 调整以满足个体约束
                    for j in range(self.N):
                        sys = self.systems[j]
                        if self.P_cmd < 0:
                            is_charging = True
                        else:
                            is_charging = False
                        available_power = sys.available_power(is_charging)
                        if is_charging:
                            if P_alloc[j] < available_power:
                                P_alloc[j] = available_power
                        else:
                            if P_alloc[j] > available_power:
                                P_alloc[j] = available_power

            # 2. 初始化粒子的各项属性
            self.positions.append(P_alloc)
            self.velocities.append(np.random.uniform(-1, 1, self.N))  # 速度
            self.pbest_positions.append(P_alloc.copy())  # 个体最优位置

            # 3. 计算适应度
            val = self._fitness(P_alloc)
            self.pbest_values.append(val)  # 个体最优适应度值

            # 4. 更新全局最优解
            if val < self.gbest_value:
                self.gbest_value = val
                self.gbest_position = P_alloc.copy()

    def _generate_random_solution(self):
        """生成满足总功率平衡的随机解"""
        P_alloc = np.zeros(self.N)

        # 计算各系统的可用功率
        if self.P_cmd < 0:
            # 充电时，各系统可用功率为正值
            is_charging=True
        else:
            # 放电时，各系统可用功率为负值
            is_charging=False
        available_powers = [sys.available_power(is_charging) for sys in self.systems] # main中初始化，然后逐一调用生成list

        # total_available考虑绝对值，认为全是正值
        total_available = sum([abs(p) for p in available_powers])

        # 情况1：可用功率不足，只能按各系统最大可用功率分配
        if total_available < abs(self.P_cmd):
            # 直接使用各系统的可用功率
            for i in range(self.N):
                P_alloc[i] = available_powers[i]

        # 情况2：可用功率充足，解的存在性肯定存在，找到一个即可，后续再用PSO优化
        elif total_available >= abs(self.P_cmd):
            # 首先生成随机比例
            ratios = np.random.dirichlet(np.ones(self.N))
            for i in range(self.N):
                P_alloc[i] = ratios[i] * self.P_cmd

            # 调整以满足个体约束，可用功率充足，则必然可以满足初始化合理分配
            for i in range(self.N):#每次调整一个系统的功率，逐一调整
                # 处理功率超限情况
                if True == is_charging:
                    # 充电时：检查是否小于最大充电功率（更加负）
                    if P_alloc[i] < available_powers[i]:
                        excess = available_powers[i] - P_alloc[i]  # 计算超出部分（正值）
                        P_alloc[i] = available_powers[i]  # 将该系统功率设为最大可用值

                        # 对除此之外的系统进行遍历，将多余充电功率分配给其他系统（需要增加其他系统的充电功率）
                        indices = [j for j in range(self.N) if j != i]
                        for j in indices:
                            if P_alloc[j] > available_powers[j]:  # 如果该系统充电功率可以增加（当前充电功率未达到最大值）
                                increasable = P_alloc[j] -available_powers[j] # 可增加的充电功率（正值）
                                increase = min(excess, increasable)  # 取更小的正值
                                P_alloc[j] = P_alloc[j] - increase  # 增加充电功率（变得更负）
                                excess = excess -increase  # 减少剩余多余功率（excess是正值）
                                if excess <= 0:  # 如果多余功率已全部分配
                                    break
                elif False == is_charging :
                    # 放电时：检查是否大于最大放电功率（更加正）
                    if P_alloc[i] > available_powers[i]:
                        excess = P_alloc[i] - available_powers[i]  # 计算超出部分（正值）
                        P_alloc[i] = available_powers[i]  # 将该系统功率设为最大可用值

                        # 将多余放电功率分配给其他系统
                        indices = [j for j in range(self.N) if j != i]
                        for j in indices:
                            if P_alloc[j] < available_powers[j]:  # 如果该系统还有剩余放电容量
                                addable = available_powers[j] - P_alloc[j]  # 可增加的放电功率（正值）
                                add = min(excess, addable)  # 取较小值
                                P_alloc[j] = P_alloc[j] + add  # 增加放电功率
                                excess = excess - add  # 减少剩余多余功率
                                if excess <= 0:  # 如果多余功率已全部分配
                                    break

        return P_alloc

    def _fitness(self, P_alloc):
        """计算适应度函数（包含惩罚项）"""
        is_charging = self.P_cmd < 0

        # 1. 计算总能量裕度（根据SOC计算，总为正值）
        E_margin_total = sum(sys.energy_margin(is_charging) for sys in self.systems)

        # 2. 计算理想分配值（根据能量裕度比例）
        if E_margin_total > 0:
            # 使用能量裕度比例分配功率，充放电方向由P_cmd的符号决定
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
        penalty += 10000 * abs(total_power - self.P_cmd) ** 2  # 增加到10000

        # 个体约束检查，硬性约束
        for i, sys in enumerate(self.systems):
            if not sys.check_constraints(P_alloc[i], self.dt):
                penalty += 100000  # 大幅增加到100000

        return fitness_val + penalty

    def solve(self):
        """运行PSO求解
        self.positions是二维列表
        形状：[num_particles][N]
        含义：包含所有粒子的位置信息
        示例：如果有20个粒子，4个储能系统，则形状为20×4
        new_position是一维数组
        形状：[N]
        含义：单个粒子更新后的位置
        示例：[sys1_power, sys2_power, sys3_power, sys4_power]
        """

        # 记录上一步的功率分配，用于平滑处理
        previous_power_allocation = None

        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social

                # 更新位置
                new_position = self.positions[i] + self.velocities[i]

                # # 硬约束处理：确保功率在允许范围内
                # for j in range(self.N):
                #     sys = self.systems[j]
                #     # 直接限制功率在[P_min, P_max]范围内
                #     self.positions[i][j] = np.clip(self.positions[i][j], sys.P_min, sys.P_max)

                # # 边界吸收处理：粒子到达边界时速度归零
                # for j in range(self.N):
                #     sys = self.systems[j]
                #     if new_position[j] < sys.P_min:
                #         new_position[j] = sys.P_min
                #         self.velocities[i][j] = 0  # 速度归零
                #     elif new_position[j] > sys.P_max:
                #         new_position[j] = sys.P_max
                #         self.velocities[i][j] = 0  # 速度归零

                # 零功率死区处理
                zero_power_threshold = 0.001  # 1W阈值
                for j in range(self.N):
                    # 如果当前功率接近0，且新位置也接近0，则强制为0
                    if abs(self.positions[i][j]) < zero_power_threshold and abs(new_position[j]) < zero_power_threshold:
                        new_position[j] = 0.0
                        self.velocities[i][j] = 0.0  # 完全停止
                    else:
                        # 正常边界吸收处理
                        if new_position[j] < self.systems[j].P_min:
                            new_position[j] = self.systems[j].P_min
                            self.velocities[i][j] = 0
                        elif new_position[j] > self.systems[j].P_max:
                            new_position[j] = self.systems[j].P_max
                            self.velocities[i][j] = 0

                self.positions[i] = new_position

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
            
            # 惯性权重衰减，后期逐步减少权重，聚焦好的方案
            self.w *= 0.99
            
            if iteration % 10 == 0:#每20次迭代输出一次
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
        previous_solution = None  # 上一步的最优解
        
        while step < self.max_steps:
            # 检查是否所有系统都无法继续
            can_continue = any(sys.can_continue(self.is_charging) for sys in self.systems)
            if not can_continue:
                print(f"所有储能系统已{'充满' if self.is_charging else '放空'}，模拟结束")
                break
            
            # 计算当前总可用功率
            total_available = sum(sys.available_power(self.is_charging) for sys in self.systems)
            total_available = abs(total_available)
            
            # 如果可用功率不足，调整指令
            effective_cmd = self.P_cmd
            if self.is_charging:
                if total_available < abs(self.P_cmd):
                    effective_cmd = -total_available  # 按最大可用充电功率
            else:
                if total_available < self.P_cmd:
                    effective_cmd = total_available  # 按最大可用放电功率
            
            # 使用PSO求解功率分配，传递上一步的最优解
            solver = PSOSolver(self.systems, effective_cmd, self.dt, num_particles=20, max_iter=50, previous_solution=previous_solution)
            best_solution, best_fitness = solver.solve()
            
            # 记录结果
            self.time_steps.append(step)
            self.total_power.append(effective_cmd)
            self.allocations.append(best_solution.copy())
            
            # 更新系统状态
            for i, sys in enumerate(self.systems):
                sys.update_SOC(best_solution[i], self.dt)
            
            # 更新上一步最优解
            previous_solution = best_solution.copy()
            
            step += 1
            
            # 每1步打印一次进度
            if step % 1 == 0:
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

            # 保持功率的实际正负值显示
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            bars = ax2.bar(system_names, powers, color=colors[:len(system_names)])
            ax2.set_xlabel('储能系统')
            ax2.set_ylabel('功率 (kW)')
            ax2.set_title(f'最后时间步功率分配 (t={time_hours[last_step]:.2f}h)')
            ax2.grid(True, alpha=0.3, axis='y')

            # 添加数值标签，显示实际功率值（包含正负号）
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}', ha='center', va='bottom' if height >= 0 else 'top')

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

                # 保持功率的实际正负值，不进行绝对值转换
                stack_data.append(sampled_power)

        if stack_data:
            # 使用实际功率值进行堆叠，正负值分别显示
            ax5.stackplot(sampled_time, stack_data, labels=[sys.name for sys in self.systems], alpha=0.8)
            ax5.set_xlabel('时间 (小时)')
            ax5.set_ylabel('功率 (kW)')
            ax5.set_title('各储能系统功率贡献（堆叠图）')
            ax5.grid(True, alpha=0.3)
            ax5.legend(loc='upper right')

            # 添加零线，便于区分充放电
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)

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
                energy_throughput = np.trapezoid(np.abs(power_array), time_array) / 3600  # kWh
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
    # # 放电
    # systems = [
    #     EnergyStorageSystem("氢燃料电池", E_rated=200, P_current=0, P_min=-60, P_max=60, SOC_current=9, SOC_min=5, SOC_max=80, dP_max=1, eff_charge=0.95, eff_discharge=0.95),
    #     EnergyStorageSystem("铁硫电池",   E_rated=100, P_current=0, P_min=-5, P_max=5,   SOC_current=6, SOC_min=5, SOC_max=90, dP_max=3, eff_charge=0.92, eff_discharge=0.92),
    #     EnergyStorageSystem("钠电池",     E_rated=100, P_current=0, P_min=-30, P_max=30, SOC_current=8, SOC_min=5, SOC_max=95, dP_max=30, eff_charge=0.98, eff_discharge=0.98),
    #     EnergyStorageSystem("钒液流电池", E_rated=100, P_current=0, P_min=-5, P_max=5,   SOC_current=7, SOC_min=5, SOC_max=85, dP_max=3,  eff_charge=0.90, eff_discharge=0.90)
    # ]
    # 充电
    systems = [
        EnergyStorageSystem("氢燃料电池", E_rated=200, P_current=0, P_min=-60, P_max=60, SOC_current=75, SOC_min=5, SOC_max=80, dP_max=1, eff_charge=0.95, eff_discharge=0.95),
        EnergyStorageSystem("铁硫电池",   E_rated=100, P_current=0, P_min=-5, P_max=5,   SOC_current=86, SOC_min=5, SOC_max=90, dP_max=3, eff_charge=0.92, eff_discharge=0.92),
        EnergyStorageSystem("钠电池",     E_rated=100, P_current=0, P_min=-30, P_max=30, SOC_current=92, SOC_min=5, SOC_max=95, dP_max=30, eff_charge=0.98, eff_discharge=0.98),
        EnergyStorageSystem("钒液流电池", E_rated=100, P_current=0, P_min=-5, P_max=5,   SOC_current=82, SOC_min=5, SOC_max=85, dP_max=3,  eff_charge=0.90, eff_discharge=0.90)
    ]
    # 电力调度功率指令（放电为正，充电为负）
    P_cmd = -100  # kW，放电指令
    
    print("开始多步长模拟...")
    print(f"调度指令: {P_cmd} kW (放电)")
    print(f"储能系统数量: {len(systems)}")
    
    # 创建多步长模拟器
    simulator = MultiStepSimulator(systems, P_cmd, dt=1, max_steps=3000)  # 时间步长5秒
    
    # 执行模拟
    steps = simulator.simulate()
    
    # 绘制结果
    simulator.plot_results()
    
    # 可选：保存结果到CSV文件
    results = simulator.get_results()
    df = pd.DataFrame(results)
    df.to_csv('energy_storage_simulation_results.csv', index=False)
    print("结果已保存到 energy_storage_simulation_results.csv")