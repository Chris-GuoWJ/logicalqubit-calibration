# LogicalQubit Calibration

Calibration and Bayesian optimization workflows for π pulses, qubit Z–frequency maps, and tunable couplers on the LogicalQubit Cloud Platform.

三个标定实验使用一致的命名和目录结构。Python 模块负责波形、测量、分析与保存；Notebook 只设置参数并调用模块。

| 目录 | 内容 |
|---|---|
| src/pi.py | π 脉冲幅度网格扫描、IQ 候选值、PiSession 单点测量 |
| src/zfreq.py | Z 偏置与驱动 IF 的频率扫描、峰值分析 |
| src/coupler.py | 双比特交换实验、联合概率和耦合强度拟合 |
| src/waveforms.py | 共用的 XY/Z 波形、8 ns 对齐、读出波形 |
| src/readout.py | raw IQ 检查和 IQ 中心距离 |
| src/runs.py | 运行目录、CSV/JSON 检查点、运行锁 |
| src/hardware.py | 恢复 LQCloud 任务 |
| src/config.py | 参数快照、配置读取、显式发布、历史版本 |
| optimize/kernels.py | RBF、Matérn 3/2、Matérn 5/2、有理二次核 |
| optimize/gp.py | GP 拟合、边缘对数似然、后验预测 |
| optimize/acquisition.py | EI 和 LCB |
| optimize/search.py | 参数范围、观测噪声、下一点选择 |
| optimize/runner.py | Pi, Z-frequency and coupler BO loops with checkpoints |

## 入口

在项目根目录或 execute 目录启动 Jupyter，然后依次使用 execute/pi.ipynb、execute/zfreq.ipynb、execute/coupler.ipynb。execute/pi_optimize.ipynb 是真实 π 幅度贝叶斯优化入口，execute/compare.ipynb 是不连接真机的核函数示例。各 Notebook 默认 RUN_HARDWARE=False；编辑范围、shots 和目标比特后再运行。

真机执行需要原环境中的 lqcloud SDK，以及 numpy、pandas、scipy、matplotlib、Jupyter。模块导入不会创建 LQCloudProvider 或提交任务。

所有实验使用 UTC 运行名 YYYYMMDDTHHMMSS_ffffffZ_PI / _ZF / _CP，按类别保存到 result/pi、result/zfreq、result/coupler。每个运行目录下面有 data/ 和 fig/。data/ 保存 manifest.json、任务 ID、原始 IQ、汇总表；fig/ 保存图片。恢复时只填 RESUME_RUN_TAG（运行目录名称），并保持原扫描设置一致。

config/calibrations 提供不含设备任务信息的示例模板。首次使用时复制相应的 `.example` 文件并移除文件名中的 `.example`，再填入目标 backend 的已验证标定参数。实际标定文件、硬件快照、运行结果和 round 记录默认不进入 Git。config/hardware 的真机属性在实际实验连接后通过 save_hardware 产生新快照。新候选参数默认留在 result；Notebook 中手动开启 PUBLISH_CONFIG 后才会更新正式配置，旧配置会备份到 config/history。

π 幅度的自动优化最小化 IQ 中心距离的相反数。该指标可用于挑选候选幅度，不等同于 RB/IRB 门错误率。当前 π 振幅、Z 频率和耦合器均已接入 BO 真机闭环，同时保留固定扫描。三种核函数的实验对比可通过 execute/compare.ipynb 和同一真机预算的重复标定完成。

离线测试（不会连接真机）：

    python -m unittest discover -s tests -v

## DeepSeek Harness agent 入口

在 DeepSeek Harness 的 Web UI 中把 `D:\SSHProgram\Platform_LQ` 选作 workspace；项目 AGENTS.md 会给 agent 提供目录、命令和实验边界。这里使用普通 shell + Python CLI，不要求先开发 Harness 插件。

在项目根目录运行：

    python agent_cli.py test
    python agent_cli.py plan config/plans/pi_bo_example.json

第二条只解析 JSON、检查参数维度及实验预算，不连接真机。检查计划后，真实运行入口为：

    python agent_cli.py run config/plans/pi_bo_example.json --execute

`run` 会连接 LQCloud，独立执行的真机任务和 shots 均受 JSON 中 `max_jobs`/`max_shots` 限制；返回运行目录，不会自动发布新标定参数。JSON 的 kind 可以是 `pi_grid`、`pi_bo`、`zfreq`、`coupler`。其中 zfreq 需要提供每个比特的 IF 扫描网格；coupler 需要两个相邻比特、耦合器 Z 网格和相互作用时间网格。恢复实验时填写 `run_mode: "resume"` 和原运行目录名 `resume_run_tag`。提出一轮新参数时保留独立的 JSON 方案，以便审核和重复。


## Adaptive BO for all three calibration modules

| JSON kind | Optimized variable | Runner | Each measurement |
| --- | --- | --- | --- |
| pi_bo | pi amplitude | optimize.runner.run_pi | Raw IQ versus one XY-off reference |
| zfreq_bo (new plan) | Select Z bias to fit peak IF f(Z) | optimize.runner.run_zfreq_map | Full drive-IF spectrum and one baseline at each proposed Z |
| zfreq_bo (legacy plan) | Drive IF at each preset Z | optimize.runner.run_zfreq | Fixed-Z raw IQ distance |
| coupler_bo | coupler Z at a mapped resonance | optimize.runner.run_coupler | Full SWAP time trace versus four joint references |

The fixed-grid kinds remain pi_grid, zfreq, and coupler. For an offline
budget check use python agent_cli.py plan config/plans/zfreq_map_bo_example.json
or python agent_cli.py plan config/plans/coupler_bo_example.json.
To submit an already reviewed plan, use python agent_cli.py run <spec.json>
--execute. The two new notebook entry points are execute/zfreq_optimize.ipynb
and execute/coupler_optimize.ipynb; they default to RUN_HARDWARE=False.

BO observations retain raw IQ, task IDs, resumable suggestions and RNG
state in result/<module>/<run>/data/. BO returns experimental candidates.
In particular, src.config.publish_z analyzes a fixed Z/IF grid and does
not publish a BO candidate frequency map. Validate peaks, SWAP fits and
gate performance independently before publishing calibration parameters.


## Live progress in hardware notebooks

Each hardware notebook under `execute/` uses `src.progress.watch_run` when
`RUN_HARDWARE=True`. The notebook displays completed/planned QPU jobs, the
current submitted job, elapsed time, and an estimated remaining time based
on jobs completed in this session. The display updates once per second from
`manifest.json`; a resumed run starts at its already saved job count, and
retries with the same label count only once. The monitor only reads checkpoint
files. It does not submit jobs or change how recovery and calibration work.
`execute/compare.ipynb` is an offline simulation and has no hardware section.


## Adaptive spectroscopy map

execute/zfreq_optimize.ipynb now uses config/plans/zfreq_map_bo_example.json.
A complete IF spectrum at each proposed Z yields one resonance-frequency
observation. A GP of f(Z) selects the next Z using posterior uncertainty.
Each Z point consumes frequency_points + 1 QPU jobs including its baseline.
The prior zfreq_bo_example.json remains a legacy fixed-Z demonstration;
it is not the new adaptive mapping workflow. Candidate maps are not
published automatically.
