# LCE Polymer Network 体积相图试验操作手册

本文档说明当前代码库中“不同液晶密度、不同温度下 polymer network 体积概率分布”的完整试验流程。目标是从同一套 3D xyz 等分取向网络出发，对多个液晶插入密度 `lcden` 和多个拓扑 seed 做降温扫描，并用 Gaussian density 或 convex hull 方法计算每个温度下 network 体积的概率分布。

## 1. 核心试验对象

本流程中的横轴“液晶密度”使用 `network.insertion_density`，也就是 eligible network segment 中被 `a-E-a` 液晶单元替换的比例。输出目录和文件名中用 `lcdenXXX` 表示 per-mille 精度：

- `0.000 -> lcden000`
- `0.125 -> lcden125`
- `0.250 -> lcden250`
- `0.750 -> lcden750`

注意：这里的 `lcden` 不是 LAMMPS 输入里旧有的 `simulation.rho`。`simulation.rho` 仍保留在配置中，但本相图试验的密度轴由 `network.insertion_density` 控制。

默认生产参数在：

```bash
/Users/joshua/Desktop/MD/polymer-network-aEa-project/params_volume_phase.json
```

默认网络骨架：

- `topology_mode = unified_lattice`
- `cells_x = cells_y = cells_z = 7`
- `x_s_count = y_s_count = z_s_count = 10`
- `axis_insertion_weights = {"x": 1, "y": 1, "z": 1}`
- `geometry_mode = preserve-grid`

这意味着 x/y/z 三个方向的液晶初始取向目标权重相等。实际插入数量会受 eligible segment 数和整数取整影响，所以每个 case 的实际 x/y/z 插入数会写入 `manifest.json` 和每个网络的 metadata。

## 2. 温度协议

现在的推荐协议是三段式 cooldown：

1. 最高温 `T*=1.40`：初始化速度到 1.40，然后在 1.40 做 `ramp -> relax -> sample`。
2. 后续温度：从上一温度 ramp 到目标温度，然后 `relax -> sample`。
3. 后处理只分析 `sample` 阶段的 dump，不把非平衡 ramp 或驰豫段混入概率分布。

当前默认温度表为降温顺序：

```text
1.40, 1.35, 1.30, 1.25, 1.20, 1.15, 1.10, 1.05, 1.00,
0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50
```

如果配置中给的是默认升序表，实验矩阵生成器会自动转换为降序 cooldown 表；如果手写 cooldown 参数，建议直接写降序，避免误解。

## 3. 你提出的最高温 hold 问题

你的判断是合理的。对于 `T*=1.40`，不应该把刚初始化后的构型直接当作采样统计。现在采用的最高温流程是：

```text
velocity create at T*=1.40
T*=1.40 ramp-equilibration
T*=1.40 relax hold
T*=1.40 sample hold
```

严格说，从 freshly generated data file 开始时，LAMMPS 没有一个低温动力学状态可以“升温到 1.40”；代码做的是在 `T*=1.40` 初始化速度，然后用同样长度的 `ramp` 阶段进行高温热化。这个阶段的作用等价于你说的“第一次升温/ramp 阶段”，但 thermostat 起止温度都是 1.40。

如果以后改成从一个低温 restart 继续跑，那么第一段可以改成真正的 `Tlow -> 1.40` 升温 ramp；目前的试验从新生成网络开始，所以用 `1.40 -> 1.40` 的高温热化更直接。

## 4. 时间长度和 dump 数

当前已采用你指定的 dump 设计：

```json
"cooldown_ramp_time_lj": "100",
"cooldown_relax_time_lj": "300",
"cooldown_sample_time_lj": "3600",
"ramp_dump_frames": 100,
"relax_dump_frames": 300,
"sample_dump_frames": 3600
```

在默认 `timestep = 0.001` 下，换算如下：

| 阶段 | LJ 时间 | LAMMPS steps | dump 帧数 | dump 间隔 |
| --- | ---: | ---: | ---: | ---: |
| ramp | 100 | 100000 | 100 | 1000 steps |
| relax | 300 | 300000 | 300 | 1000 steps |
| sample | 3600 | 3600000 | 3600 | 1000 steps |

因此每个温度目录理论上会产生：

- 100 个 `ramp` dump frame
- 300 个 `relax` dump frame
- 3600 个 `sample` dump frame

对体积概率分布来说，默认只读取 `sample` dump。`ramp` 和 `relax` 文件保留用于诊断温度、体积是否达到稳定，以及检查是否出现异常转变或数值不稳定。

重启文件当前按 `restart_interval_dump_frames` 控制。生产配置里是：

```json
"restart_interval_dump_frames": 10
```

由于 `DTSample = 1000 steps`，所以 restart 频率为 `10000 steps`。

## 5. 每个温度的 LAMMPS 输出结构

每个 case 的结果目录类似：

```text
experiments/volume_phase_diagram/cases/lcden250_seed12345/result/
  Tstar_1.40/
  Tstar_1.35/
  ...
  Tstar_0.50/
```

每个温度文件夹包含三类轨迹：

```text
HEAV.lcden250.ramp.*.dump
HEAV.lcden250.relax.*.dump
HEAV.lcden250.sample.*.dump
```

采样阶段还会输出：

```text
volume_state.lcden250.sample.dat
TOPOLOGY.lcden250.data
Final.cooldown.lcden250.bin
Restart.cooldown.lcden250.*
log.Tstar_1.40.sample.lammps
```

`volume_state` 记录 box 体积和粒子坐标包络范围，适合快速检查 LAMMPS box volume 与后处理 network volume 是否趋势一致。真正的概率分布以后处理 dump 的 Gaussian density 或 convex hull 体积为准。

## 6. 生成试验矩阵

先创建实验矩阵：

```bash
cd /Users/joshua/Desktop/MD/polymer-network-aEa-project
python3 scripts/create_volume_phase_experiments.py \
  --params params_volume_phase.json \
  --suite-dir experiments/volume_phase_diagram
```

这会生成：

```text
experiments/volume_phase_diagram/manifest.json
experiments/volume_phase_diagram/cases/<lcden>_<seed>/params.json
experiments/volume_phase_diagram/cases/<lcden>_<seed>/inputs/
experiments/volume_phase_diagram/cases/<lcden>_<seed>/result/
```

默认密度矩阵：

```text
0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750
```

默认 seed：

```text
12345, 22345, 32345
```

默认总 case 数为 `7 densities x 3 seeds = 21 cases`。

## 7. 运行 LAMMPS

正式运行前先 dry-run：

```bash
python3 scripts/run_experiment_matrix.py \
  --manifest experiments/volume_phase_diagram/manifest.json \
  --dry-run
```

确认命令和路径正确后再运行：

```bash
python3 scripts/run_experiment_matrix.py \
  --manifest experiments/volume_phase_diagram/manifest.json
```

运行器会逐个 case 调用每个 case 自己的 `params.json`，并把输出写入该 case 的 `result/` 目录。每个 case 是隔离的，不会共用 dump 或 restart 文件名。

服务器上验证两个 seed、每个 case 使用 `np4_omp2` 时，可以直接用：

```bash
cd /Users/joshua/Desktop/MD/polymer-network-aEa-project
./scripts/server_run_two_seed_np4_omp2.sh
```

这个脚本默认运行 7 个 `lcden` 和 2 个 seed，共 14 个 case；每个 case 使用：

```text
mpiexec -np 4 lmp_mpi
OMP_NUM_THREADS=2
```

也就是每个 case 占用 `4 x 2 = 8` 个 CPU。默认 `JOBS=14`，所以总 CPU 使用量是 `14 x 8 = 112`。如果服务器给了 256 CPU，这个两 seed 验证矩阵是够用的；理论上最多可同时跑 `256 / 8 = 32` 个 case。

如果服务器上的 LAMMPS 命令不同，可以用环境变量覆盖：

```bash
MPIEXEC=mpirun LAMMPS_BIN=/path/to/lmp CPU_TOTAL=256 JOBS=14 \
  ./scripts/server_run_two_seed_np4_omp2.sh
```

如果只想生成输入和检查命令，不启动 LAMMPS：

```bash
DRY_RUN_ONLY=1 ./scripts/server_run_two_seed_np4_omp2.sh
```

如果服务器上希望每个 case 放进独立 tmux session，并且所有输出写到你当前 `cd` 的目录，而不是写到代码目录，用：

```bash
cd /your/server/run/directory
/path/to/polymer-network-aEa-project/scripts/server_tmux_two_seed_np4_omp2.sh
```

这个 tmux 脚本仍然默认运行 14 个 case，但会启动 14 个 tmux session：

```text
lce_np4omp2_lcden000_seed12345
lce_np4omp2_lcden000_seed22345
...
lce_np4omp2_lcden750_seed22345
```

输出目录在当前 `cd` 目录下：

```text
experiments/volume_phase_np4_omp2_2seed_tmux/
  manifest.json
  cases/<case_id>/inputs/
  cases/<case_id>/result/
  tmux/logs/<case_id>.tmux.log
  tmux/status/<case_id>.status
```

查看状态：

```bash
experiments/volume_phase_np4_omp2_2seed_tmux/tmux/check_tmux_status.sh
```

进入某个 case：

```bash
tmux attach -t lce_np4omp2_lcden250_seed12345
```

只生成输入和检查命令，不启动 tmux：

```bash
DRY_RUN_ONLY=1 /path/to/polymer-network-aEa-project/scripts/server_tmux_two_seed_np4_omp2.sh
```

如果你想让脚本一直等到所有 tmux session 结束，并在全部成功后自动做 Gaussian volume analysis：

```bash
WAIT_FOR_FINISH=1 RUN_ANALYSIS=1 \
  /path/to/polymer-network-aEa-project/scripts/server_tmux_two_seed_np4_omp2.sh
```

如果上一次同名 tmux session 还存在，需要重启同一批 session：

```bash
REPLACE_EXISTING=1 /path/to/polymer-network-aEa-project/scripts/server_tmux_two_seed_np4_omp2.sh
```

## 8. 体积分析

默认分析 `sample` 阶段：

```bash
python3 scripts/analyze_volume_probability.py \
  experiments/volume_phase_diagram \
  --method gaussian \
  --grid-spacing 0.5 \
  --bins 50
```

如果只想分析每个温度最后一部分采样帧，可以加：

```bash
--tail-frames-per-tstar 1000 --stride 1
```

如果计算量太大，可以先用：

```bash
--tail-frames-per-tstar 1000 --stride 5
```

Gaussian density 参数默认：

- `grid_spacing = 0.5`
- `threshold = exp(-0.5)`
- `gaussian_cutoff_q2 = 6.0`
- `selected_types = {1, 2}`，即 ellipsoid 和 sphere；anchor 默认不参与体积。

Convex hull 方法：

```bash
python3 scripts/analyze_volume_probability.py \
  experiments/volume_phase_diagram \
  --method convex_hull \
  --bins 50
```

Convex hull 需要安装 SciPy；Gaussian density 只依赖标准库。

## 9. 分析输出

默认输出目录：

```text
experiments/volume_phase_diagram/analysis/volume_probability/
```

主要文件：

```text
Volume_probability_summary.dat
Volume_probability_bins_all_cases.dat
Volume_phase_diagram.dat
```

`Volume_probability_summary.dat`：每个 case、每个温度的体积均值、标准差、中位数、min/max、block mean 置信区间。

`Volume_probability_bins_all_cases.dat`：每个 case、每个温度的体积概率分布直方图。

`Volume_phase_diagram.dat`：按 `lcden_tag` 和 `Tstar` 聚合多个 seed 的平均体积与 seed 间 SEM，用于画相图。

## 10. 建议的质量检查

每次正式跑生产矩阵前，至少检查：

1. `manifest.json` 中每个 `lcden` 的 `actual_lc_insertion_fraction` 是否接近目标值。
2. `axis_insertions` 是否接近 x/y/z 等分。
3. 每个温度目录是否都有 `ramp`、`relax`、`sample` 三类 dump。
4. `log.Tstar_*.sample.lammps` 中温度是否在目标温度附近波动。
5. `volume_state.*.sample.dat` 中 box volume 是否没有数值爆炸或非物理突跳。
6. 后处理 summary 的 `sample_count` 是否符合预期。
7. 不要把 `ramp` 或 `relax` 阶段混入最终概率分布，除非目的是诊断非平衡路径。

## 11. 当前方案取舍

当前方案选择 cooldown 而不是每个温度独立初始化，原因是你要看的是同一个 network 随温度降低的体积响应，cooldown 可以减少初始构相差异对相图的干扰。

最高温也拆成 `ramp/relax/sample`，原因是刚生成的网络和刚初始化的速度不应该直接进入概率统计。`ramp` 阶段负责热化，`relax` 阶段负责让体积和局部取向进一步驰豫，`sample` 阶段才进入概率分布。

输出 dump 数采用 `100/300/3600`，原因是它让每个阶段的 dump 间隔在默认 timestep 下都等于 1000 steps，便于比较和调试；同时 sample 阶段明显长于 ramp/relax，统计权重集中在平衡后的体积分布上。
