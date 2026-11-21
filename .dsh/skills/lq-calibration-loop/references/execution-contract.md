# 固定执行接口与数据契约

## 命令

从 `D:\SSHProgram\Platform_LQ` 运行：

```powershell
python agent_cli.py plan config/plans/pi_q9_round02.json
python agent_cli.py run config/plans/pi_q9_round02.json --execute
python .dsh/skills/lq-calibration-loop/scripts/summarize_run.py result/pi/<run_tag>
```

前两条调用现有受信执行器，第三条只读已保存 BO 数据。
不要用 Notebook 运行代替 CLI；不要假定存在未实现的 CLI 命令。
`plan` 不连接硬件，不检查完整硬件拓扑、最新校准有效性、全部波形或目录可写性。
`run` 会连接 LQCloud、写硬件快照并执行有限计划；CLI 返回 plan、run_dir、run_status 和候选摘要。
代码升级/异常数据结构交给维护者处理，不递归研究并修改实现。

## 计划字段

保留相同 kind 模板中的字段；只修改本次任务允许改变的字段。

| kind | 主要字段 | 新 run 计划 jobs |
|---|---|---|
| pi_bo | qubit, bounds, steps, initial_points | 1 + steps |
| pi_grid | qubits, amplitudes（按比特 ID 字符串索引） | Σq(1 + 振幅点数q) |
| zfreq_bo | qubit, z_values, bounds（IF GHz）, steps, initial_points | Z 点数 × (1 + steps) |
| zfreq | qubits, z_values, frequencies（按比特 ID 索引） | Z 点数 × Σq(1 + IF 点数q) |
| coupler_bo | qubits（有序）, reference_z, tuning_z_range, tuning_z_hint, bounds, times_ns, steps, initial_points | 4 + steps × 时间点数 |
| coupler | qubits（有序）, reference_z, tuning_z_range, tuning_z_hint, coupler_z_values, times_ns | 4 + coupler Z 点数 × 时间点数 |

共用 backend、shots、max_jobs、max_shots、settings、run_mode、resume_run_tag。
每轮 shots 总数为 jobs × shots；参考任务也消耗预算。
steps 包含初始探索点，不是“初始点数 + 后续点数”；设 initial_points >= 2 且 steps > initial_points，
以确保确有 GP 选点。pi_bo 虽然当前 preflight 较宽松，也遵循这个规则。
支持 kernel=rbf/matern32/matern52/rq，acquisition=ei/lcb。保持既有核，除非专门比较核。
kappa 仅影响 lcb；ei 下改变 kappa 不会改变选点策略。不要虚构 CLI 没有暴露的 xi 参数。
相同 seed 不能替代独立实验复测。

轴需有限、严格递增；π 振幅非负；数字 IF 严格在 (-0.5, 0.5) GHz 内，
该采样限制不是设备允许的全部物理范围。耦合器时间轴至少 8 点，且为整数 ns。
不要为满足 RAM 预检任意削减 relaxation/readout；settings 默认继承已审核设置。
coupler 的 qubits[0] 为 tuning qubit、qubits[1] 为 reference qubit；不要随意调换顺序。
bounds/tuning_z_range/reference_z 控制不同物理量，不可互换。

## 恢复与计费

新实验设 run_mode=new、resume_run_tag=""。
恢复设 run_mode=resume、resume_run_tag 为精确目录名，保留原计划/预算/设置。
恢复运行的预算包括先前尝试；提交状态不明的任务仍可能已消耗额度。
整任务累计预算必须覆盖所有轮次、参考、验证及重试；
当前 CLI 不负责跨轮次账本，不允许代理自行把 max_jobs/max_shots 调高以绕过授权。

## 结果文件

读取 `result/<pi|zfreq|coupler>/<run_tag>/data/manifest.json`：
先查 kind、strategy、backend、比特、shots、settings、plan、jobs、status。
任务 label/job_id/status 是恢复依据，不能修改它们。
原始 IQ 为 npz；CSV 可能有 UTF-8 BOM，使用 utf-8-sig 读取。
优先读小型汇总文件，只有诊断需要时再查看对应原始 IQ。

| 实验 | 文件与语义 |
|---|---|
| pi BO | observations.csv：amplitude、S、loss、noise_variance、shots、job_id、baseline_job_id |
| zfreq BO | observations.csv：按 z_index/z_amplitude 分组的 drive_if_ghz、S、loss、noise_variance；frequency_candidates.csv 为候选摘要 |
| coupler BO | observations.csv：coupler_z、score、loss、p01_contrast、anti_corr、noise_variance、shots、job_ids、reference_job_ids |
| coupler 时间曲线 | joint_probabilities.csv：coupler_z、duration_ns、本地修正的 p00/p01/p10/p11 及未修正分类概率 *_raw；readout_calibration.json 保存四参考矩阵 |
| 固定网格 | 用对应 manifest 确认实际保存布局；不要强行按 BO CSV 解释 |

coupler BO 的每行 shots 是整条时间曲线的 shots 总和，区别于计划中每个 job 的 shots。
图像输出在对应 run 的 fig/。已有图优先只读。
维护者可使用 optimize.show_fig.plot_bo_run 生成 BO 诊断图，但此调用可能写图或补写摘要，
不把它当作“纯只读”操作，也不把写权限开放给任意临时分析脚本。
