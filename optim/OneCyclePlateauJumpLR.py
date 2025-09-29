import math
from torch.optim.lr_scheduler import _LRScheduler
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import SGD


class OneCyclePlateauJumpLR(_LRScheduler):
    """
    基于 OneCycleLR 扩展的学习率调度器，支持多周期执行：
      - 初始与结束学习率可控：div_factor、final_div_factor
      - 上升/下降阶段的停滞（plateau）策略：均匀、等差、指数、对数
      - 下降阶段的学习率跳跃（jump）功能
      - 多周期循环（num_cycles）：后续周期的 max_lr 按比例衰减
      - up_steps 也随着周期增加进行0.5衰减

    曲线示例（num_cycles=2）：
      周期1: initial_lr → max_lr → … → final_lr
      周期2: final_lr → max_lr/2 → … → final_lr

    参数：
        optimizer (Optimizer): 需调度的优化器
        max_lr (float): 第一周期学习率峰值
        total_steps (int): 每周期总迭代步数
        pct_start (float): 上升阶段所占比例 (0< pct_start <1)
        div_factor (float): 初始学习率 = max_lr / div_factor
        final_div_factor (float): 最终学习率 = max_lr / final_div_factor

        plateau_up_count (int): 上升阶段停滞次数
        plateau_up_steps (int): 每次停滞的步数
        plateau_up_strategy (str): 上升阶段停滞分布策略，选项 ['uniform','arithmetic','exponential','log']

        plateau_down_count (int): 下降阶段停滞次数
        plateau_down_steps (int): 每次停滞的步数
        plateau_down_strategy (str): 下降阶段停滞分布策略，选项同上

        num_jumps (int): 下降阶段跳跃次数
        jump_magnitude (float): 跳跃倍率，在跳跃点 lr *= jump_magnitude
        anneal_strategy (str): 插值策略，'linear' 或 'cos'

        num_cycles (int): 总周期数（默认为1）
        cycle_decay (float): 后续周期 max_lr 衰减倍率（默认为0.5，即减半）
        last_epoch (int): 起始 epoch，默认为 -1
    """
    def __init__(self,
                 optimizer,
                 max_lr,
                 total_steps,
                 pct_start=0.3,
                 div_factor=25,
                 final_div_factor=1e4,
                 plateau_up_count=1,
                 plateau_up_steps=100,
                 plateau_up_strategy='uniform',
                 plateau_down_count=1,
                 plateau_down_steps=200,
                 plateau_down_strategy='uniform',
                 num_jumps=1,
                 jump_magnitude=1.1,
                 jump_once_steps=200,
                 anneal_strategy='linear',
                 num_cycles=1,
                 cycle_decay=0.5,
                 last_epoch=-1):
        # 基本参数
        self.base_max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy
        self.last_lr = self.base_max_lr / self.div_factor

        # 停滞配置
        self.plateau_up_count = plateau_up_count
        self.plateau_up_steps = plateau_up_steps
        self.plateau_up_strategy = plateau_up_strategy
        self.plateau_down_count = plateau_down_count
        self.plateau_down_steps = plateau_down_steps
        self.plateau_down_strategy = plateau_down_strategy

        # 跳跃配置
        self.num_jumps = num_jumps
        self.jump_magnitude = jump_magnitude
        self.jump_once_steps = jump_once_steps

        # 多周期配置
        self.num_cycles = max(1, int(num_cycles))
        self.sigle_cycles_steps = total_steps // self.num_cycles
        self.cycle_decay = cycle_decay
        self.current_cycle = 0  # 从第0周期开始

        # 初始化当前周期的 max/initial/final lr
        self._reset_cycle_lrs()

        super().__init__(optimizer, last_epoch)

    def _reset_cycle_lrs(self):
        """根据 current_cycle 计算本周期初始、峰值、结束学习率"""
        # 衰减后的 max_lr
        self.max_lr = self.base_max_lr * (self.cycle_decay ** self.current_cycle)
        # 初始与结束
        self.initial_lr = self.max_lr / self.div_factor
        self.final_lr = self.base_max_lr / self.final_div_factor  # final_lr 恒定不变

        # 分阶段步数
        self.up_steps = int(math.floor(self.sigle_cycles_steps * self.pct_start * (1 / (self.current_cycle + 1))))
        self.down_steps = self.sigle_cycles_steps - self.up_steps

        # 计算停滞位置
        self.plateau_up_positions = self._compute_segments(
            phase_steps=self.up_steps,
            count=self.plateau_up_count,
            seg_len=self.plateau_up_steps,
            offset=0,
            strategy=self.plateau_up_strategy
        )
        self.plateau_down_positions = self._compute_segments(
            phase_steps=self.down_steps,
            count=self.plateau_down_count,
            seg_len=self.plateau_down_steps,
            offset=self.up_steps,
            strategy=self.plateau_down_strategy
        )
        # 计算跳跃点
        if self.num_jumps > 0 and self.down_steps > 0:
            total_jump_steps = self.jump_once_steps * self.num_jumps
            if total_jump_steps >= self.down_steps:
                raise ValueError("跳跃总时长超过下降阶段，请调整 jump_once_steps 或 num_jumps")
            # 预留出 jump 段后，再平分剩余空间作为间隔
            free_space = self.down_steps - total_jump_steps
            interval = free_space // (self.num_jumps + 1)
            # 计算 jump 开始点（每次 jump 的起点）
            self.jump_steps = [
                self.up_steps + interval * (i + 1) + self.jump_once_steps * i
                for i in range(self.num_jumps)
            ]
        else:
            self.jump_steps = []

    def _compute_segments(self, phase_steps, count, seg_len, offset, strategy):
        """按策略计算停滞起始位置集合"""
        positions = set()
        if count <= 0 or phase_steps <= seg_len * count:
            return positions
        total_available = phase_steps - seg_len * count

        if strategy == 'uniform':
            interval = total_available // (count + 1)
            for i in range(count):
                start = offset + interval * (i + 1)
                positions.update(range(start, start + seg_len))
        else:
            if strategy == 'arithmetic':
                weights = [i + 1 for i in range(count)]
            elif strategy == 'exponential':
                weights = [2 ** i for i in range(count)]
            elif strategy == 'log':
                weights = [math.log(i + 2) for i in range(count)]
            else:
                raise ValueError(f"不支持的停滞分布策略: {strategy}")
            sum_w = sum(weights)
            cumulative = 0
            for w in weights:
                seg_offset = int(total_available * w / sum_w)
                start = offset + cumulative + seg_offset
                positions.update(range(start, start + seg_len))
                cumulative += seg_offset + seg_len
        return positions

    def _anneal(self, start, end, pct):
        """线性或余弦插值"""
        if self.anneal_strategy == 'cos':
            return end + (start - end) * (math.cos(math.pi * pct) + 1) / 2
        return start + pct * (end - start)

    def get_lr(self):
        step = self.last_epoch
        # 尚未开始
        if step < 0:
            return [self.initial_lr for _ in self.base_lrs]

        # 周期内步数
        local_step = step % self.sigle_cycles_steps

        # 上升阶段
        if local_step <= self.up_steps:
            if local_step in self.plateau_up_positions:
                lr = self.last_lr
            else:
                pct = local_step / float(max(1, self.up_steps))
                lr = self._anneal(self.initial_lr, self.max_lr, pct)
        else:
            # 下降阶段
            down_step = local_step - self.up_steps
            if local_step in self.plateau_down_positions:
                lr = self.last_lr
            else:
                pct = down_step / float(max(1, self.down_steps))
                lr = self._anneal(self.max_lr, self.final_lr, pct)
                
                # 跳跃
                if local_step in self.jump_steps:
                    lr *= self.jump_magnitude

        return [lr for _ in self.base_lrs]

    def step(self, epoch=None):
        """
        更新到下一个 step，每 sigle_cycles_steps 步，切换到下一个周期并重置 lr 参数
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        # 检测周期边界
        if epoch // self.sigle_cycles_steps > self.current_cycle and self.current_cycle + 1 < self.num_cycles:
            self.current_cycle += 1
            # 重置下一个周期的 lr 参数
            self._reset_cycle_lrs()
        self.last_epoch = epoch
        self.last_lr = self.get_lr()[0]
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.last_lr

            
if __name__ == '__main__':

    # 构造一个模型和优化器
    model = nn.Linear(10, 2)
    optimizer = SGD(model.parameters(), lr=0.1)

    # 使用调度器
    scheduler = OneCyclePlateauJumpLR(
        optimizer=optimizer,
        max_lr=0.01,
        total_steps=100,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=1e3,
        plateau_up_count=2,
        plateau_up_steps=2,
        plateau_down_count=2,
        plateau_down_steps=2,
        num_jumps=2,
        jump_magnitude=1.5,
        jump_once_steps=2,
        anneal_strategy='cos',
        num_cycles=1,
        cycle_decay=1
    )

    # 模拟 step
    lr_list = []
    for step in range(100):
        scheduler.step()
        lr_list.append(optimizer.param_groups[0]['lr'])

    # 可视化
    plt.plot(lr_list)
    plt.title("OneCyclePlateauJumpLR")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid()
    plt.show()
