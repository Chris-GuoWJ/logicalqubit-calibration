# Optimization coverage

kernels.py, gp.py, acquisition.py, and search.py provide the same
Gaussian-process optimizer to three independent measurement sessions.

| BO entry | Measurement session | One BO observation | Candidate output |
| --- | --- | --- | --- |
| run_pi | src.pi.PiSession | One raw-IQ distance against XY-off reference | Pi amplitude |
| run_zfreq_map | src.zfreq_map.ZMapSession | One IF spectrum at a proposed Z bias | GP frequency map f(Z) |
| run_zfreq (legacy) | src.zfreq.ZSession | One IQ distance at a fixed Z bias | Drive IF for each preset Z |
| run_coupler | src.coupler.CouplerSession | A full fixed-time SWAP trace, after four joint references | Coupler Z |

run_zfreq_map sweeps IF at each actively chosen Z bias, locates an interior
spectral peak, and fits one GP to peak drive IF as a function of Z.
Initial Z values span the requested range; subsequent values maximize GP
posterior uncertainty away from sampled Z values. A spectral branch
still needs independent validation. The new plan is
config/plans/zfreq_map_bo_example.json: steps Z points and frequency_points
IF measurements per Z, for steps * (frequency_points + 1) jobs.
New runs save observations.csv, frequency_candidates.csv and
frequency_map_posterior.csv. Legacy preset-Z BO runs and their prior plans
remain readable. Fixed-grid analyze and publish_z reject both BO strategies.

run_coupler fits one GP over coupler Z at a fixed, mapped tuning-qubit Z.
Its loss is minus (corrected P01 time contrast times nonnegative P10/P01
anticorrelation). Each candidate collects the same time grid. Shot-noise
variance is bootstrapped conditional on this run's reference assignment matrix;
it does not cover drift or uncertainty in that matrix. Inspect fitted SWAP
traces and validate the gate separately before publishing parameters.

All three loops save raw IQ, server job IDs, a pending suggestion, and
optimizer RNG state. A missing job ID is treated as an ambiguous submission
and requires server reconciliation. Result folders contain data/ and fig/.
There is no automatic promotion of BO candidates to config/calibrations.

Examples: config/plans/pi_bo_example.json,
config/plans/zfreq_map_bo_example.json, and
config/plans/coupler_bo_example.json. agent_cli.py plan <path> previews
jobs/shots without connecting to LQCloud. agent_cli.py run <path> --execute
submits real hardware jobs; review waveform limits and source calibrations
before using any example. Existing zfreq_grid_example.json and
coupler_grid_example.json still run fixed grids.


## BO figure replay

Call `optimize.show_fig.plot_bo_run(run_dir)` from any notebook to load an
existing BO run, display the saved GP evolution and final fit, and save the
figures under that run's `fig/` directory. It never submits jobs. The three
`execute/*_optimize.ipynb` notebooks use this function; set `PLOT_RUN_TAG`
in the Z-frequency or coupler notebook to replay a previous run offline.

The new Z-frequency map draws an evolving GP of f(Z) alongside
measured spectra. Legacy fixed-Z BO runs still draw a separate GP for
each requested Z.

Coupler plots the actual BO selection score (SWAP contrast times nonnegative
anticorrelation), with its bootstrapped noise variance. New coupler BO runs
save these pointwise values to `data/observations.csv` alongside the
time-resolved `data/joint_probabilities.csv`. The fit's conditional LOO
scores keep hyperparameters fitted on all points and are diagnostics, not
independent validation or gate fidelity.

