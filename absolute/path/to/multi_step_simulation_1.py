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
        
        # 如果是充电工况，将功率转换为正值显示
        if self.is_charging:
            powers = np.abs(powers)  # 充电功率取绝对值
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax2.bar(system_names, powers, color=colors[:len(system_names)])
        ax2.set_xlabel('储能系统')
        
        # 根据充放电调整y轴标签
        if self.is_charging:
            ax2.set_ylabel('充电功率 (kW)')
            ax2.set_title(f'最后时间步充电功率分配 (t={time_hours[last_step]:.2f}h)')
        else:
            ax2.set_ylabel('放电功率 (kW)')
            ax2.set_title(f'最后时间步放电功率分配 (t={time_hours[last_step]:.2f}h)')
        
        ax2.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
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
            
            # 如果是充电工况，将功率转换为正值显示
            if self.is_charging:
                sampled_power = np.abs(sampled_power)  # 充电功率取绝对值
            stack_data.append(sampled_power)
    
    if stack_data:
        # 如果是充电工况，调整堆叠方向
        if self.is_charging:
            # 充电时从下往上堆叠
            ax5.stackplot(sampled_time, stack_data, labels=[sys.name for sys in self.systems], alpha=0.8)
            ax5.set_ylabel('充电功率 (kW)')  # 修改y轴标签
        else:
            # 放电时正常堆叠
            ax5.stackplot(sampled_time, stack_data, labels=[sys.name for sys in self.systems], alpha=0.8)
            ax5.set_ylabel('放电功率 (kW)')
        
        ax5.set_xlabel('时间 (小时)')
        ax5.set_title('各储能系统功率贡献（堆叠图）')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()