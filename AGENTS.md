# Platform_LQ agent guide

## Calibration operator tasks

For planning, running and analyzing calibration experiments, load the project skill
`lq-calibration-loop` at `.dsh/skills/lq-calibration-loop/SKILL.md`.
Use its experiment-specific references and fixed CLI contract.
Start from the current plan and the exact prior run; do not recursively explore
the code architecture or run the development test suite for each calibration task.

Directly edit only new files under `config/plans/`, including round records under
`config/plans/notes/`. Keep prior plans immutable once submitted.
Treat `agent_cli.py`, `src/`, `optimize/`, `execute/`, tests and skill files as
fixed implementation. Keep `config/calibrations/` read-only.

- Preview with `python agent_cli.py plan config/plans/<round>.json`.
  This checks plan structure and job/shot counts; it is not a complete hardware dryrun.
- Submit with `python agent_cli.py run config/plans/<round>.json --execute`
  only within a user-authorized finite experiment task and parameter envelope.
  Authorization may cover multiple rounds; do not request it again for each authorized round.
- Do not treat model-editable max_jobs/max_shots as an independent authorization.
  Track the total task budget across exploration, references, validation and retries.
- The runner must write new data/checkpoints under result/ and new snapshots under
  config/hardware/. A plans-only process write policy cannot run the current CLI.
  Skills constrain behavior; strict separation requires a separately trusted executor.
- Read the exact manifest and saved data under result/<kind>/<run>/data/.
  Never overwrite earlier runs or manually edit their manifests.
- Restore the original plan with run_mode=resume and the exact resume_run_tag.
  Reconcile unknown submissions with the server before submitting anything new.
- Keep backend, ordered qubits, units, reviewed settings and parameter limits explicit.
  Digital IF is GHz; time is ns; platform amplitudes are not automatically volts.
- IQ distance, exchange score and fitted coupling strength are candidate signals,
  not RB/IRB gate fidelity. Record independent validation; do not auto-publish candidates.
- Stop on filesystem/SDK/credential failures and report the exact error.
  Do not repair permissions, expose secrets or change execution code during an experiment.
- Do not execute experiment notebooks unattended or run concurrent experiments by default.

## Explicit code-maintenance tasks

When the user specifically requests implementation changes, read README.md and the
relevant source files, make scoped changes and run appropriate offline checks with
`python agent_cli.py test`. Do not submit hardware jobs as part of development checks.
Keep credentials in their normal local configuration and retain existing run data.
