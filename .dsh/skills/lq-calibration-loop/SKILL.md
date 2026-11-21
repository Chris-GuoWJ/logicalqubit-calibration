---
name: lq-calibration-loop
description: 在 Platform_LQ 中仅编辑 config/plans 内的计划和轮次记录，调用固定 CLI 运行已授权的超导量子标定实验，读取结果并调整下一轮计划。用于 π 振幅、Z 频率和耦合器的计划—测量—分析闭环；不要求通读或修改实验源码。
---

# Platform_LQ 标定闭环

## 角色与最小上下文

担任实验规划与分析代理。把已审核 Python 执行器当作固定仪器接口。
默认只读本文件、当前计划、目标 run 的 manifest/汇总数据、相关标定与硬件快照。
按实验类型读取下列一份策略；不要递归浏览源码、Notebook、全部历史数据或反复跑全套开发测试。

- π 振幅：读取 [pi-strategy.md](references/pi-strategy.md)。
- Z 频率：读取 [zfreq-strategy.md](references/zfreq-strategy.md)。
- 耦合器：读取 [coupler-strategy.md](references/coupler-strategy.md)。
- 首次执行或不确定字段/预算：读取 [execution-contract.md](references/execution-contract.md)。
- 配置权限或遇到拒绝访问：读取 [permissions.md](references/permissions.md)。

仅直接编辑项目 `config/plans/` 中的新计划和 `config/plans/notes/` 中的轮次记录。
保持旧计划、旧 run、正式标定文件不变；不要编辑本 skill、源码、测试、Notebook 或凭据。
通过固定 CLI 触发的结果/硬件快照写入属于执行器职责。该约定不等于操作系统强制隔离。

## 建立本次实验任务边界

从当前用户任务或管理员只读政策获取：backend、比特、允许改变的参数及硬边界、
每轮和整个任务的 jobs/shots/时间/轮数预算，以及停止与独立验证要求。
当前样例计划不是新的真机授权；“编写/安装 skill”也不是运行实验的授权。
在用户已明确授权的有限任务范围内连续推进，无需每轮重复询问。
缺失的授权只阻断提交，继续完成数据分析和下一轮草案。
不把计划内代理可编辑的 `max_jobs/max_shots` 当作独立的硬授权，也不自行增加整个任务预算。
把一次任务的授权、剩余预算、历史 run 和下一轮目的写入轮次记录。
硬边界和预算的强制执行需要独立受信策略/执行服务；当前 CLI 只有单 run 预算控制。

## 每轮流程

1. 明确上一轮 run 的精确路径、backend、比特、策略、标定来源和时间；新任务允许没有上一轮。
2. 先读 manifest 检查状态及未确认提交。对已有 BO run 使用只读摘要：
   `python .dsh/skills/lq-calibration-loop/scripts/summarize_run.py result/<kind>/<run_tag>`。
   摘要不导入 LQCloud、不重算原始实验、不写结果。
3. 读取相关策略。区分实测最优、模型预测最优、已独立验证的参数。
   检查噪声、边界峰、谱峰/时间曲线、漂移和预算，提出一个可检验的下一轮目的。
4. 从相同类型的模板复制为新的轮次 JSON，例如
   `config/plans/pi_q9_round02.json`；把理由、父 run、变更前后值、
   预期信息、实际消耗和停止条件写入 `config/plans/notes/pi_q9_round02.json`。
   不用当前全部最优值覆盖旧模板，不把 notes JSON 当作实验计划运行。
5. 调用 `python agent_cli.py plan config/plans/<round>.json`。
   该命令只检查部分字段与预算；不要称其为完整波形/硬件/权限 dryrun。
   CLI 当前没有 `dryrun`、`analyze` 或 `status` 子命令。
6. 在授权、硬边界、总预算和执行器写入权限均满足时，调用
   `python agent_cli.py run config/plans/<round>.json --execute`。
   同一硬件任务串行执行；不要并发启动两个 run 或重启尚在运行的命令。
   硬件提交前固定本轮计划内容，运行期间只读它。
7. 以返回的准确 run_dir 为准，结合 manifest 跟踪已保存任务、待处理 job_id、
   花费时间和终止状态；约每 30–60 秒提供简短反馈。
   不用“最新目录”猜测本轮数据，不因无 stdout 就判定卡死。
8. 完成后分析本轮；按停止条件结束，或在剩余授权内进入下一轮。
   记录停止原因、候选与不确定度、独立验证结果、尚待解决的问题。

## 数据与统计纪律

使用 manifest 中的单位与设置。drive IF 的单位为 GHz；时间为 ns；Z/XY 振幅沿用平台单位，
不要擅自解释为伏特、绝对 RF 频率或归一化门角。
BO 内部 `loss=-S` 或 `loss=-score`，因此 loss 越小、S/score 越大越好。
把 `sqrt(noise_variance)` 作为当前噪声模型下的误差尺度；不要视为全部系统误差。
共享参考测量会导致不同点的误差相关，coupler bootstrap 还条件于参考矩阵。
用同参数独立复测检验小幅改进；不凭最优单点噪声或 GP 峰直接宣称成功。
比较跨 run 数据时核对 shots、波形、读出设置、标定来源、backend 和漂移。
条件 LOO 的 RMSE 保留 score 单位；95% coverage 只是模型诊断，
固定全数据超参数的 LOO 不等于独立测试，也不是门保真度。
不得把待验证候选自动发布到 `config/calibrations`。

## 故障与恢复

拒绝访问、缺凭据、SDK/波形错误属于执行问题；记录具体路径和错误后停止提交并交给维护者。
不要在标定任务中修权限、改源码、读密钥、安装依赖或扩大写权限。
对于网络超时，先读 manifest；有 job_id 则核对服务器状态并按原规格恢复。
对未确认提交保持停止，不把它自动当作失败重新提交。
`resume` 只恢复相同计划、shots、设置和精确 run_tag；改参数要创建 new run。
预算耗尽、预设精度达到、改进未能复测成立、硬边界限制或数据质量不合格时停止。
明确报告“停止取数”与“标定验证通过”是不同结论。
