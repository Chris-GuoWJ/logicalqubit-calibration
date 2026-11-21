"""LQCloud amplitude sweep: XY-off reference, raw IQ, maximum center distance.

Wave conventions (1 ns samples, GHz IF, negative complex carrier) follow the
project's z_frequency_calibration.ipynb. No cloud calls occur during import.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
from time import monotonic, sleep, time_ns
from contextlib import contextmanager
import os
import tempfile
import numpy as np
from .runs import (new_run, resume_run, data_dir, fig_dir,
                   save_json as _json, load_manifest as _load_manifest,
                   save_csv as _csv, run_lock as _run_lock)
from .readout import extract_iq, center_distance as iq_statistics
from .hardware import restore_job as _restore_job
from .waveforms import xy_wave, readout


@dataclass(frozen=True)
class PulseSettings:
    duration_ns: int = 40
    relaxation_ns: int = 80_000
    read_delay_ns: int = 40
    tail_ns: int = 1000
    envelope: str = "raised_cosine"
    # The target backend reports a 120 us RAM capacity. Reserve is an engineering margin,
    # not an exact model of the server's additional waveform allocations.
    ram_limit_ns: int = 120_000
    ram_reserve_ns: int = 30_000

    def validate(self):
        for name in ("duration_ns", "relaxation_ns", "read_delay_ns", "tail_ns",
                     "ram_limit_ns", "ram_reserve_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer in ns")
        if self.duration_ns < 4 or self.relaxation_ns == 0:
            raise ValueError("duration_ns >= 4 and relaxation_ns > 0 required")
        if self.envelope not in ("raised_cosine", "square"):
            raise ValueError("envelope must be raised_cosine or square")
        if self.ram_limit_ns <= self.ram_reserve_ns:
            raise ValueError('ram_limit_ns must exceed ram_reserve_ns')


def waveform_budget(qconf, settings):
    """Conservative local budget; final RAM allocation is determined by the QPU."""
    settings.validate()
    length = qconf['read_length']
    if not np.isfinite(length) or int(length) != length or length < 8:
        raise ValueError('read_length must be an integer >= 8 ns')
    total = ((settings.relaxation_ns + settings.duration_ns + settings.read_delay_ns
              + int(length) + settings.tail_ns + 7) // 8) * 8
    budgeted = total + settings.ram_reserve_ns
    if budgeted > settings.ram_limit_ns:
        raise ValueError(
            f'Waveform RAM preflight failed: waveform={total / 1000:.3f} us + '
            f'reserve={settings.ram_reserve_ns / 1000:.3f} us > '
            f'limit={settings.ram_limit_ns / 1000:.3f} us. '
            'Reduce relaxation_ns or other waveform durations; do not increase the '
            'hardware limit without device confirmation. Long shot waits require '
            'a platform repetition/idle mechanism outside waveform RAM.')
    return {'waveform_ns': total, 'reserved_ns': settings.ram_reserve_ns,
            'budgeted_ns': budgeted, 'limit_ns': settings.ram_limit_ns}


def build_plan(qid, qconf, amplitude, drive_if_ghz, settings):
    """Every shot starts with passive relaxation, then XY, then one readout.

    XY-off reference uses the exact same timing and readout with amplitude=0.
    No Z bias or active-reset waveform is added. Idle approaches thermal ground
    state, not a guaranteed pure |0>. Confirm wait and device waveform limits.
    """
    settings.validate()
    values = [amplitude, drive_if_ghz, qconf['read_amp'], qconf['df_read'], qconf['read_length']]
    if not np.isfinite(values).all() or amplitude < 0:
        raise ValueError("Pulse/readout values must be finite; amplitude >= 0")
    if abs(drive_if_ghz) >= .5 or abs(qconf['df_read']) >= .5:
        raise ValueError("At 1 ns sampling, IF must be strictly within +/-0.5 GHz")
    read_len = int(qconf['read_length'])
    if read_len != qconf['read_length'] or read_len < 8:
        raise ValueError("read_length must be an integer >= 8 ns")
    start = settings.relaxation_ns
    read_start = start + settings.duration_ns + settings.read_delay_ns
    total = waveform_budget(qconf, settings)['waveform_ns']
    xy = xy_wave(total, start, settings.duration_ns, amplitude, drive_if_ghz,
                 envelope=settings.envelope)
    return {f'q{qid}.xy': [xy], f'q{qid}.z': [np.zeros(total, dtype=np.float32)],
            **readout(qid, qconf, total, read_start)}














def _load_raw(path, amplitude, frequency, shots):
    """Validate durable rawIQ instead of trusting a possibly stale completed flag."""
    try:
        with np.load(path, allow_pickle=False) as data:
            iq = data['iq'].copy()
            if iq.shape != (shots, 2) or not np.isfinite(iq).all():
                raise ValueError('invalid rawIQ shape or nonfinite data')
            if float(data['amplitude']) != amplitude or float(data['drive_if_ghz']) != frequency:
                raise ValueError('amplitude/frequency does not match the scan plan')
            job_id = str(data['job_id'].item())
            if not job_id:
                raise ValueError('missing job ID')
            baseline_id = str(data['baseline_job_id'].item()) if 'baseline_job_id' in data else None
        return iq, job_id, baseline_id
    except Exception as exc:
        raise ValueError(f'Resume refused: invalid checkpoint {path}: {exc}') from exc




def _validate_resume(saved, expected):
    if saved.get('schema_version', 1) not in (1, 2):
        raise ValueError('Unsupported resume manifest version')
    fields = ('backend', 'settings', 'shots', 'qubits', 'frequency_unit',
              'sample_interval_ns', 'readout_correction', 'baseline')
    different = [key for key in fields if saved.get(key) != expected[key]]
    if list(saved.get('qubits', {})) != list(expected['qubits']):
        different.append('qubit order')
    if saved.get('waveform_version', 'pi_wave_v1') != 'pi_wave_v1':
        different.append('waveform_version')
    if different:
        raise ValueError('Resume refused because scan settings changed: ' + ', '.join(different)
                         + '. Restore the saved parameters or use run_mode="new".')


def _checkpoint_keys(run_dir, manifest, scans, shots):
    completed = set()
    latest = {(e['qubit'], e['label']): e for e in manifest['jobs']}
    for qid, (amps, frequency) in scans.items():
        labels = {'baseline': 0., **{f'point_{i:04d}': float(a) for i, a in enumerate(amps)}}
        directory = data_dir(run_dir) / f'q{qid}'
        baseline_id = None
        for path in directory.glob('*.npz'):
            if path.stem not in labels:
                raise ValueError(f'Unexpected checkpoint file: {path}')
        for label, amplitude in labels.items():
            path = directory / f'{label}.npz'
            if not path.exists():
                continue
            _, job_id, reference_id = _load_raw(path, amplitude, frequency, shots)
            if label == 'baseline':
                baseline_id = job_id
            elif reference_id is not None and baseline_id is not None and reference_id != baseline_id:
                raise ValueError(f'Baseline job ID mismatch: {path}')
            entry = latest.get((qid, label))
            if entry is None or entry.get('job_id') != job_id:
                raise ValueError(f'Checkpoint job ID does not match manifest: {path}')
            entry['status'] = 'completed'  # Handles interruption after NPZ rename.
            completed.add((qid, label))
    return completed


def run_calibration(backend, qubit_metrics, amplitudes_by_qubit, output_root,
                    settings=None, drive_ifs=None, shots=1000, timeout=900,
                    show_plots=False, run_mode='new', resume_run_tag='',
                    job_loader=None, config_dir=None):
    """Submit sequential jobs; save each raw result before the next submission.

    Returns (run_directory, candidate_list). This function executes real jobs
    when given a real backend. Resume reuses rawIQ and queries recorded job IDs;
    only server-confirmed FAILED/CANCELLED attempts are replaced on resume.
    job_loader(job_id) is an optional adapter for tests or other SDK versions.
    """
    import matplotlib.pyplot as plt  # Check plotting dependency before submitting jobs.
    run_mode = str(run_mode).strip().lower()
    if run_mode not in ('new', 'resume'):
        raise ValueError('run_mode must be "new" or "resume"')
    root = Path(output_root).resolve()
    if run_mode == 'resume':
        tag = str(resume_run_tag).strip()
        if not tag or tag in ('.', '..') or '/' in tag or '\\' in tag or ':' in tag:
            raise ValueError('resume_run_tag must be a single existing run directory name')
        run_dir = resume_run(root, tag, 'pi')
    elif str(resume_run_tag).strip():
        raise ValueError('Clear resume_run_tag when run_mode="new"')
    settings = settings or PulseSettings()
    settings.validate()
    if isinstance(shots, bool) or not isinstance(shots, int) or not 2 <= shots <= 50_000:
        raise ValueError('shots must be an integer between 2 and 50000')
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive')
    if not amplitudes_by_qubit:
        raise ValueError('No target qubits')
    drive_ifs = drive_ifs or {}
    scans = {}
    for qid, amplitudes in amplitudes_by_qubit.items():
        if isinstance(qid, bool) or not isinstance(qid, int) or qid < 0:
            raise ValueError('Qubit IDs must be nonnegative integers')
        amps = np.asarray(amplitudes, dtype=float)
        if amps.ndim != 1 or len(amps) < 3 or not np.isfinite(amps).all() or np.any(amps < 0) or np.any(np.diff(amps) <= 0):
            raise ValueError('Use >= 3 finite, nonnegative, strictly increasing amplitudes')
        q = qubit_metrics[qid]
        frequency = float(drive_ifs.get(qid, q['df_10']))
        build_plan(qid, q, float(amps[-1]), frequency, settings)
        budget = waveform_budget(q, settings)
        print(f'Q{qid}: waveform={budget["waveform_ns"] / 1000:.3f} us, '
              f'with reserve={budget["budgeted_ns"] / 1000:.3f} us, '
              f'RAM limit={budget["limit_ns"] / 1000:.3f} us', flush=True)
        scans[qid] = (amps, frequency)
    total_jobs = sum(1 + len(amps) for amps, _ in scans.values())
    expected = {'status': 'running', 'settings': asdict(settings), 'shots': shots,
                'schema_version': 2, 'waveform_version': 'pi_wave_v1',
                'use_async': bool(getattr(backend, 'use_async', True)),
                'backend': str(getattr(backend, 'name', 'unknown')),
                'frequency_unit': 'GHz IF, NOT absolute RF', 'sample_interval_ns': 1,
                'readout_correction': False, 'baseline': 'passive relaxation, XY off',
                'S_definition': 'norm(mean(IQ_at_amplitude) - mean(baseline_IQ))',
                'qubits': {str(q): {'amplitudes': a.tolist(), 'drive_if_ghz': f,
                    'read_amp': float(qubit_metrics[q]['read_amp']),
                    'df_read': float(qubit_metrics[q]['df_read']),
                    'read_length': float(qubit_metrics[q]['read_length'])} for q, (a, f) in scans.items()},
                'waveform_budgets': {str(q): waveform_budget(qubit_metrics[q], settings) for q in scans},
                'total_jobs': total_jobs, 'completed_jobs': 0, 'jobs': []}
    if run_mode == 'new':
        run_dir = new_run(root, 'pi')
    with _run_lock(run_dir):
        if run_mode == 'resume':
            manifest = _load_manifest(run_dir)
            _validate_resume(manifest, expected)
        else:
            manifest = expected
        result = _execute_scan(backend, qubit_metrics, scans, settings, shots, timeout,
                               show_plots, run_dir, manifest, total_jobs, run_mode, job_loader)
        if config_dir is not None:
            try:
                export_pi_amplitudes(run_dir, config_dir)
            except Exception:
                print('测量已完成，但 config 导出失败；可单独调用 export_pi_amplitudes 重试，无需重测。', flush=True)
                raise
        return result


def export_pi_amplitudes(run_dir, config_dir):
    """Export a complete run; update only its qubits in pi_pulse_amplitudes.json.

    Recompute maxima and readout centers from validated rawIQ so partial/stale
    summary files cannot be published. No hardware submission occurs here.
    """
    run_dir = Path(run_dir).resolve()
    manifest = _load_manifest(run_dir)
    if manifest.get('status') != 'completed':
        raise ValueError('Config export requires a completed run')
    scans = {int(q): (np.asarray(v['amplitudes'], dtype=float), float(v['drive_if_ghz']))
             for q, v in manifest['qubits'].items()}
    shots = manifest['shots']
    expected_count = sum(len(amps) + 1 for amps, _ in scans.values())
    if not scans or len(_checkpoint_keys(run_dir, manifest, scans, shots)) != expected_count:
        raise ValueError('Config export refused: rawIQ checkpoints are incomplete')
    updates = {}
    stamp = datetime.now(timezone.utc).isoformat()
    for qid, (amps, frequency) in scans.items():
        directory = data_dir(run_dir) / f'q{qid}'
        baseline, baseline_id, _ = _load_raw(directory / 'baseline.npz', 0., frequency, shots)
        rows = []
        for index, amplitude in enumerate(amps):
            iq, job_id, _ = _load_raw(directory / f'point_{index:04d}.npz', float(amplitude), frequency, shots)
            rows.append({**iq_statistics(iq, baseline), 'job_id': job_id})
        best_index = int(np.argmax([r['S'] for r in rows]))
        best = rows[best_index]
        updates[str(qid)] = {
            'pi_amp': float(amps[best_index]), 'drive_if_ghz': frequency,
            'pulse_settings': manifest['settings'], 'S_max': best['S'],
            'iq_centers': {
                'state_0': {'I': best['I0'], 'Q': best['Q0'],
                            'job_id': baseline_id, 'shots': shots,
                            'preparation': 'initial_XY_off_baseline'},
                'state_1': {'I': best['I_mean'], 'Q': best['Q_mean'],
                            'job_id': best['job_id'], 'shots': shots,
                            'preparation': 'pulse_at_selected_pi_amplitude'}},
            'readout_settings': {
                'read_amp': manifest['qubits'][str(qid)]['read_amp'],
                'df_read_ghz': manifest['qubits'][str(qid)]['df_read'],
                'read_length_ns': manifest['qubits'][str(qid)]['read_length'],
                'integration_length_ns': min(int(manifest['qubits'][str(qid)]['read_length']) // 8 * 8, 4000),
                'readout_correction': manifest['readout_correction']},
            'at_scan_boundary': best_index in (0, len(amps) - 1),
            'low_signal': best['S'] <= 3 * best['mean_displacement_noise'],
            'status': 'candidate_requires_first_Rabi_peak_check',
            'source_run_tag': run_dir.name, 'source_run_dir': str(run_dir),
            'job_id': best['job_id'], 'baseline_job_id': baseline_id,
            'shots': shots, 'updated_at_utc': stamp}
    config_dir = Path(config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / 'pi_pulse_amplitudes.json'
    with _run_lock(config_dir):
        config = {'schema_version': 1, 'backend': manifest['backend'],
                  'selection_rule': 'argmax norm(mean(IQ_A) - mean(baseline_IQ))',
                  'amplitude_unit': 'platform waveform units', 'qubits': {}}
        if path.exists():
            config = json.loads(path.read_text(encoding='utf-8'))
            if (config.get('schema_version') != 1 or config.get('backend') != manifest['backend']
                    or not isinstance(config.get('qubits'), dict)):
                raise ValueError('Existing pi_pulse_amplitudes.json has an incompatible schema/backend')
        config['qubits'].update(updates)
        config['updated_at_utc'] = stamp
        _json(path, config)
    message = f'config 更新完成（π 振幅及 0/1 态 IQ 读取中心）：{path}；更新比特：{", ".join("Q" + q for q in updates)}'
    print(message, flush=True)
    with (data_dir(run_dir) / 'progress.log').open('a', encoding='utf-8') as handle:
        handle.write(f'[{stamp}] {message}\n')
    return path


def _execute_scan(backend, qubit_metrics, scans, settings, shots, timeout,
                  show_plots, run_dir, manifest, total_jobs, run_mode, job_loader):
    import matplotlib.pyplot as plt
    started = monotonic()
    completed_keys = _checkpoint_keys(run_dir, manifest, scans, shots)
    completed_jobs = len(completed_keys)

    def report(message):
        line = f'[{datetime.now().astimezone().strftime("%H:%M:%S")}] {message}'
        print(line, flush=True)
        with (data_dir(run_dir) / 'progress.log').open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    report(f'{"恢复" if run_mode == "resume" else "开始"}扫描：已保存 {completed_jobs}/{total_jobs} 个测量任务；'
           f'每任务 {shots} shots；RUN_TAG={run_dir.name}；结果目录：{run_dir}')
    if run_mode == 'resume':
        report('复用原 0 态基准；若长时间中断或设备重新标定导致漂移，请使用 new 重新采集。')
    manifest.update(status='running', completed_jobs=completed_jobs, total_jobs=total_jobs)
    manifest.pop('error_type', None)
    manifest.setdefault('sessions', []).append({'mode': run_mode, 'started_utc': datetime.now(timezone.utc).isoformat()})
    _json(data_dir(run_dir) / 'manifest.json', manifest)
    candidates = []
    try:
        for qid, (amps, frequency) in scans.items():
            directory = data_dir(run_dir) / f'q{qid}'
            directory.mkdir(exist_ok=True)

            def acquire(amplitude, label):
                nonlocal completed_jobs
                task_started = monotonic()
                task_number = completed_jobs + 1
                task_label = '0 态基准' if label == 'baseline' else f'振幅点 {int(label.split("_")[1]) + 1}/{len(amps)}'
                raw_path = directory / f'{label}.npz'
                if (qid, label) in completed_keys:
                    iq, job_id, reference_id = _load_raw(raw_path, amplitude, frequency, shots)
                    if label != 'baseline' and reference_id is not None and reference_id != baseline_id:
                        raise ValueError(f'Baseline job ID mismatch: {raw_path}')
                    report(f'[跳过 {completed_jobs}/{total_jobs}] Q{qid} {task_label}，A={amplitude:.6g}，复用 rawIQ，job_id={job_id}')
                    return iq, job_id
                entry = next((e for e in reversed(manifest['jobs']) if e['qubit'] == qid and e['label'] == label), None)
                job = None
                if entry is not None:
                    if not entry.get('job_id'):
                        raise RuntimeError(f'Q{qid} {label}: previous submission has no recorded job ID. '
                                           'Check platform task history and restore its job_id in manifest.json; '
                                           'submission outcome is unknown, so automatic resubmission is refused.')
                    report(f'[{task_number}/{total_jobs}] 查询原任务 job_id={entry["job_id"]}，Q{qid} {task_label}')
                    job = (job_loader(entry['job_id']) if job_loader else
                           _restore_job(backend, entry['job_id'], manifest.get('use_async', bool(getattr(backend, 'use_async', True)))))
                    status = job.status()
                    status_name = str(getattr(status, 'name', status)).upper()
                    entry['server_status'] = status_name
                    _json(data_dir(run_dir) / 'manifest.json', manifest)
                    if status_name in ('FAILED', 'CANCELLED'):
                        # A reference already used by saved measurements must not be replaced.
                        if label == 'baseline' and any(q == qid and lab != 'baseline' for q, lab in completed_keys):
                            raise RuntimeError('Original baseline rawIQ is missing and its job cannot be recovered; use a new run.')
                        report(f'原任务 {entry["job_id"]} 已确认 {status_name}；本次恢复将重新提交这个未完成点')
                        entry['status'] = status_name.lower()
                        job = None
                    elif status_name not in ('COMPLETED', 'RUNNING', 'QUEUED', 'PENDING', 'INITIALIZING'):
                        raise RuntimeError(f'Unknown server status {status_name}; no duplicate task submitted')
                    else:
                        report(f'原任务状态 {status_name}，继续获取同一 job_id 的结果，不重复提交')
                if job is None:
                    if label == 'baseline' and any(q == qid and lab != 'baseline' for q, lab in completed_keys):
                        raise RuntimeError('Original baseline is missing; cannot mix saved points with a new baseline')
                    report(f'[{task_number}/{total_jobs}] 正在提交 Q{qid} {task_label}，A={amplitude:.6g}')
                    entry = {'qubit': qid, 'label': label, 'amplitude': float(amplitude), 'status': 'submitting'}
                    manifest['jobs'].append(entry)
                    _json(data_dir(run_dir) / 'manifest.json', manifest)
                    job = backend.run_wave(build_plan(qid, qubit_metrics[qid], amplitude, frequency, settings),
                                           shots=shots, result_format='raw', readout_correction=False)
                    entry.update(job_id=str(job.job_id), status='submitted')
                    _json(data_dir(run_dir) / 'manifest.json', manifest)
                    report(f'[{task_number}/{total_jobs}] 已提交，job_id={job.job_id}；等待真机结果')
                try:
                    iq = extract_iq(job.result(timeout=timeout, verbose=False), qid, shots)
                except Exception as exc:
                    entry.update(status='result_error', error_type=type(exc).__name__, error_message=str(exc))
                    _json(data_dir(run_dir) / 'manifest.json', manifest)
                    raise
                temporary = raw_path.with_suffix('.npz.tmp')
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, iq=iq, amplitude=amplitude,
                                        drive_if_ghz=frequency, job_id=str(job.job_id),
                                        baseline_job_id=str(job.job_id) if label == 'baseline' else baseline_id)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(raw_path)
                entry['status'] = 'completed'
                entry['elapsed_seconds'] = round(monotonic() - task_started, 3)
                completed_jobs += 1
                completed_keys.add((qid, label))
                manifest['completed_jobs'] = completed_jobs
                _json(data_dir(run_dir) / 'manifest.json', manifest)
                center = iq.mean(axis=0)
                report(f'[{completed_jobs}/{total_jobs}，{completed_jobs / total_jobs:.1%}] 测量完成 '
                       f'Q{qid} {task_label}，A={amplitude:.6g}，'
                       f'I={center[0]:.6g}，Q={center[1]:.6g}，'
                       f'本任务耗时 {entry["elapsed_seconds"]:.1f}s；rawIQ 已保存：{directory / (label + ".npz")}')
                return iq, str(job.job_id)

            baseline, baseline_id = acquire(0., 'baseline')
            _json(directory / 'baseline.json', {'job_id': baseline_id, 'shots': shots,
                  'I0': float(baseline[:, 0].mean()), 'Q0': float(baseline[:, 1].mean())})
            print(f'Q{qid}: baseline saved, center={baseline.mean(axis=0)}', flush=True)
            baseline_fig = plot_iq_point(run_dir, qid)
            report(f'Q{qid} 基准 IQ 图{_plot_save_note(baseline_fig)}')
            if show_plots:
                plt.show()
            plt.close(baseline_fig)
            rows = []
            for index, amplitude in enumerate(amps):
                iq, job_id = acquire(float(amplitude), f'point_{index:04d}')
                rows.append({'qubit': qid, 'point_index': index, 'amplitude': float(amplitude),
                             'drive_if_ghz': frequency, 'shots': shots, 'job_id': job_id,
                             'baseline_job_id': baseline_id, **iq_statistics(iq, baseline)})
                _csv(directory / 'scan.csv', rows)
                report(f'Q{qid} 振幅点 {index + 1}/{len(amps)}：S={rows[-1]["S"]:.6g}；正在生成图像')
                point_fig = plot_iq_point(run_dir, qid, index)
                signal_fig = plot_signal(run_dir, qid)
                report(f'Q{qid} 本轮处理完成：IQ 图{_plot_save_note(point_fig)}；S 曲线{_plot_save_note(signal_fig)}')
                if show_plots:
                    plt.show()
                plt.close(point_fig)
                plt.close(signal_fig)
            best_index = int(np.argmax([row['center_distance'] for row in rows]))
            best = rows[best_index]
            candidate = {'qubit': qid, 'pi_amplitude_candidate': best['amplitude'],
                         'drive_if_ghz': frequency, **asdict(settings),
                         'center_distance': best['center_distance'],
                         'at_scan_boundary': best_index in (0, len(rows) - 1),
                         'low_signal': best['center_distance'] <= 3 * best['mean_displacement_noise'],
                         'status': 'candidate_requires_first_Rabi_peak_check',
                         'job_id': best['job_id']}
            candidates.append(candidate)
            _json(directory / 'candidate.json', candidate)
            _csv(data_dir(run_dir) / 'pi_amplitude_candidates.csv', candidates)
            report(f'Q{qid} 扫描完成：候选 π 振幅={best["amplitude"]:.6g}，'
                   f'S_max={best["center_distance"]:.6g}，'
                   f'边界峰={candidate["at_scan_boundary"]}，低信号={candidate["low_signal"]}')
        manifest['status'] = 'completed'
        report(f'全部完成：{completed_jobs}/{total_jobs} 个任务，累计耗时 {monotonic() - started:.1f}s；结果目录：{run_dir}')
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error_type=type(exc).__name__)
        last = manifest['jobs'][-1] if manifest['jobs'] else {}
        report(f'扫描中断：已完成 {completed_jobs}/{total_jobs} 个测量任务；'
               f'最近任务 Q{last.get("qubit", "?")} {last.get("label", "?")}，'
               f'job_id={last.get("job_id", "尚未取得")}，阶段={last.get("status", "unknown")}；'
               f'{type(exc).__name__}: {exc}；已保存结果：{run_dir}')
        raise
    finally:
        manifest['sessions'][-1]['elapsed_seconds'] = round(monotonic() - started, 3)
        _json(data_dir(run_dir) / 'manifest.json', manifest)
    return run_dir, candidates


def _save_plot(fig, path):
    """Write a unique PNG first; keep it if Windows refuses to replace the target.

    Plot filesystem errors are nonfatal: rawIQ/CSV persistence remains strict.
    The returned figure records the actual path (or None on save failure).
    """
    path = Path(path)
    temporary = None
    fig.calibration_saved_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', prefix=path.stem + '__',
                                         suffix='.png', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            fig.savefig(handle, format='png', dpi=160)
        try:
            temporary.replace(path)
            fig.calibration_saved_path = path
            return
        except OSError as exc:
            # The complete, closed PNG is already durable under its unique name.
            fig.calibration_saved_path = temporary
            message = f'图像主文件无法更新（可能被占用）：{path}；本轮图已另存：{temporary}；{exc}'
    except OSError as exc:
        message = f'图像保存失败，继续实验；rawIQ/CSV 可用于之后重画：{path}；{exc}'
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    print(message, flush=True)
    # Logging must not turn a nonessential PNG failure into a scan failure.
    try:
        with (path.parents[2] / 'data' / 'progress.log').open('a', encoding='utf-8') as handle:
            handle.write(f'[{datetime.now().astimezone().isoformat()}] {message}\n')
    except OSError:
        pass


def _plot_save_note(fig):
    path = getattr(fig, 'calibration_saved_path', None)
    return f'已保存：{path}' if path is not None else '未保存，可从 rawIQ/CSV 重画'


def plot_iq_point(run_dir, qid, point_index=None):
    """Save all shots for one amplitude (or baseline); return a figure for display."""
    import matplotlib.pyplot as plt
    directory = data_dir(run_dir) / f'q{qid}'
    figures = fig_dir(run_dir) / f'q{qid}'
    figures.mkdir(exist_ok=True)
    with np.load(directory / 'baseline.npz') as data:
        baseline = data['iq'].copy()
    zero = baseline.mean(axis=0)
    label = 'baseline' if point_index is None else f'point_{point_index:04d}'
    with np.load(directory / f'{label}.npz') as data:
        iq = data['iq'].copy()
        amplitude = float(data['amplitude'])
        frequency = float(data['drive_if_ghz'])
    center = iq.mean(axis=0)
    signal = float(np.linalg.norm(center - zero))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(baseline[:, 0], baseline[:, 1], s=9, alpha=.25, color='gray', label='Baseline shots')
    if point_index is not None:
        ax.scatter(iq[:, 0], iq[:, 1], s=9, alpha=.35, color='tab:blue', label=f'Measured shots (n={len(iq)})')
        ax.plot([zero[0], center[0]], [zero[1], center[1]], color='tab:red', linewidth=1)
        ax.scatter(*center, s=180, marker='*', color='tab:red', edgecolors='black', linewidths=.5, zorder=5, label='Measured mean IQ')
    ax.scatter(*zero, s=85, marker='x', color='black', linewidths=2, zorder=6, label='Baseline mean IQ')
    ax.set(xlabel='I', ylabel='Q', aspect='equal',
           title=f'Q{qid}: {label}, A={amplitude:.6g}\nIF={frequency:+.6f} GHz, S={signal:.6g}')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_plot(fig, figures / f'{label}_iq.png')
    return fig


def plot_signal(run_dir, qid):
    """Save the S-versus-amplitude curve for all acquired points, including partial scans."""
    import matplotlib.pyplot as plt
    directory = data_dir(run_dir) / f'q{qid}'
    figures = fig_dir(run_dir) / f'q{qid}'
    figures.mkdir(exist_ok=True)
    with (directory / 'scan.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    amplitudes = np.array([float(r['amplitude']) for r in rows])
    signals = np.array([float(r['center_distance']) for r in rows])
    best = int(np.argmax(signals))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(amplitudes, signals, 'o-', label='Measured S')
    ax.axvline(amplitudes[best], color='tab:red', linestyle='--', label=f'Current max: A={amplitudes[best]:.6g}')
    ax.set(xlabel='XY peak amplitude / platform units', ylabel='S = distance from baseline mean IQ',
           title=f'Q{qid}: S versus pulse amplitude ({len(rows)} acquired points)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    _save_plot(fig, figures / 's_vs_amplitude.png')
    return fig


def plot_qubit(run_dir, qid):
    """Plot all IQ means and the distance curve; keep raw shots in NPZ files."""
    import matplotlib.pyplot as plt
    directory = data_dir(run_dir) / f'q{qid}'
    figures = fig_dir(run_dir) / f'q{qid}'
    figures.mkdir(exist_ok=True)
    with (directory / 'scan.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    amplitudes = np.array([float(r['amplitude']) for r in rows])
    distances = np.array([float(r['center_distance']) for r in rows])
    centers = np.array([[float(r['I_mean']), float(r['Q_mean'])] for r in rows])
    baseline = np.load(directory / 'baseline.npz')['iq']
    zero = baseline.mean(axis=0)
    best = int(np.argmax(distances))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(amplitudes, distances, 'o-')
    axes[0].axvline(amplitudes[best], color='red', linestyle='--', label=f'candidate: {amplitudes[best]:.6g}')
    axes[0].set(xlabel='XY peak amplitude / platform units', ylabel='S = distance from baseline mean IQ', title=f'Q{qid}: amplitude sweep')
    axes[0].legend()
    axes[1].scatter(baseline[:, 0], baseline[:, 1], s=4, alpha=.12, color='gray', label='baseline shots')
    dots = axes[1].scatter(centers[:, 0], centers[:, 1], c=amplitudes, cmap='viridis', label='mean IQ')
    axes[1].scatter(*zero, marker='x', c='black', s=100, label='baseline mean')
    axes[1].scatter(*centers[best], marker='*', c='red', s=150, label='selected mean')
    axes[1].set(xlabel='I', ylabel='Q', title='IQ centers at every amplitude', aspect='equal')
    axes[1].legend()
    fig.colorbar(dots, ax=axes[1], label='Amplitude')
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    _save_plot(fig, figures / 'calibration.png')
    return fig


class PiSession:
    """One-qubit adaptive pi-amplitude measurements with durable raw IQ.

    The objective is minus IQ-center distance, not gate infidelity.
    Keep this result a candidate until an independent gate validation passes.
    """

    def __init__(self, backend, qid, qconf, output_root, plan, settings=None,
                 shots=1000, drive_if_ghz=None, run_mode="new",
                 resume_run_tag="", timeout=900, job_loader=None):
        self.backend, self.qid, self.qconf = backend, int(qid), dict(qconf)
        self.settings = settings or PulseSettings()
        self.shots, self.timeout = shots, timeout
        self.frequency = float(qconf["df_10"] if drive_if_ghz is None else drive_if_ghz)
        self.job_loader = job_loader
        if isinstance(shots, bool) or not isinstance(shots, int) or not 2 <= shots <= 50000:
            raise ValueError("shots must be an integer between 2 and 50000")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        build_plan(self.qid, self.qconf, 0., self.frequency, self.settings)
        expected = {"schema_version": 1, "kind": "pi", "strategy": "bo",
                    "backend": str(backend.name), "qubit": self.qid,
                    "shots": shots, "drive_if_ghz": self.frequency,
                    "settings": asdict(self.settings), "qconf": {
                        key: float(qconf[key]) for key in ("read_amp", "df_read", "read_length")},
                    "plan": plan, "jobs": [], "status": "running"}
        if run_mode == "new":
            if resume_run_tag:
                raise ValueError("Clear resume_run_tag for a new run")
            self.run = new_run(output_root, "pi")
            self.manifest = expected
            _json(data_dir(self.run) / "manifest.json", self.manifest)
        elif run_mode == "resume":
            self.run = resume_run(output_root, resume_run_tag, "pi")
            self.manifest = _load_manifest(self.run)
            for key, value in expected.items():
                if key not in ("jobs", "status") and self.manifest.get(key) != value:
                    raise ValueError(f"Resume refused because {key} differs")
        else:
            raise ValueError("run_mode must be new or resume")
        (data_dir(self.run) / f"q{self.qid}").mkdir(exist_ok=True)

    def _save(self):
        _json(data_dir(self.run) / "manifest.json", self.manifest)

    def _path(self, label):
        return data_dir(self.run) / f"q{self.qid}" / f"{label}.npz"

    def _acquire(self, label, amplitude, baseline_id=None):
        started = monotonic()
        path = self._path(label)
        if path.is_file():
            iq, job_id, saved_baseline = _load_raw(path, amplitude, self.frequency, self.shots)
            if baseline_id is not None and saved_baseline != baseline_id:
                raise ValueError(f"Baseline job ID mismatch: {path}")
            return iq, job_id
        entry = next((e for e in reversed(self.manifest["jobs"])
                      if e["label"] == label), None)
        if label == "baseline" and any(e["label"] != "baseline" and
                                       self._path(e["label"]).is_file()
                                       for e in self.manifest["jobs"]):
            raise RuntimeError("Cannot replace a baseline already used by saved points")
        if entry is not None:
            if float(entry["amplitude"]) != amplitude:
                raise ValueError("Pending amplitude differs from the saved request")
            if not entry.get("job_id"):
                raise RuntimeError("Submission status unknown; inspect server before resuming")
            job = (self.job_loader(entry["job_id"]) if self.job_loader else
                   _restore_job(self.backend, entry["job_id"]))
            status_obj = job.status()
            status = str(getattr(status_obj, "name", status_obj)).upper()
            if status in ("FAILED", "CANCELLED"):
                entry["status"] = status.lower()
                self._save()
                job = None
            elif status not in ("COMPLETED", "DONE", "SUCCESS", "RUNNING", "QUEUED", "PENDING", "INITIALIZING"):
                raise RuntimeError(f"Unknown server status: {status}")
        else:
            job = None
        if job is None:
            entry = {"label": label, "qubit": self.qid,
                     "amplitude": float(amplitude), "job_id": None,
                     "submitted_utc": datetime.now(timezone.utc).isoformat()}
            self.manifest["jobs"].append(entry)
            self._save()
            job = self.backend.run_wave(
                build_plan(self.qid, self.qconf, amplitude, self.frequency, self.settings),
                shots=self.shots, result_format="raw", readout_correction=False)
            entry["job_id"] = str(job.job_id)
            self._save()
        iq = extract_iq(job.result(timeout=self.timeout, verbose=False), self.qid, self.shots)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, iq=iq, amplitude=amplitude,
                                drive_if_ghz=self.frequency, job_id=str(job.job_id),
                                baseline_job_id=baseline_id or str(job.job_id))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        entry["status"] = "saved"
        entry["elapsed_seconds"] = round(monotonic() - started, 3)
        self._save()
        return iq, str(job.job_id)

    def observations(self):
        baseline_path = self._path("baseline")
        if not baseline_path.is_file():
            if any(e["label"] != "baseline" for e in self.manifest["jobs"]):
                raise ValueError("Points exist but the baseline is missing")
            return []
        reference, baseline_id, _ = _load_raw(baseline_path, 0., self.frequency, self.shots)
        records = []
        labels = sorted({e["label"] for e in self.manifest["jobs"] if e["label"].startswith("bo_")})
        for label in labels:
            entry = next(e for e in reversed(self.manifest["jobs"]) if e["label"] == label)
            path = self._path(label)
            if not path.is_file():
                continue
            amplitude = float(entry["amplitude"])
            iq, job_id, linked_baseline = _load_raw(path, amplitude, self.frequency, self.shots)
            if linked_baseline != baseline_id:
                raise ValueError(f"Wrong baseline for {path}")
            stat = iq_statistics(iq, reference)
            records.append({"amplitude": amplitude, "loss": -stat["S"],
                            "noise_variance": stat["mean_displacement_noise"] ** 2,
                            "shots": self.shots, "job_id": job_id,
                            "baseline_job_id": baseline_id,
                            "elapsed_seconds": float(entry.get("elapsed_seconds", 0)), **stat})
        return records

    def measure(self, amplitude):
        amplitude = float(amplitude)
        build_plan(self.qid, self.qconf, amplitude, self.frequency, self.settings)
        with _run_lock(self.run):
            self.manifest = _load_manifest(self.run)
            _, baseline_id = self._acquire("baseline", 0.)
            pending = [e for e in self.manifest["jobs"]
                       if e["label"].startswith("bo_") and
                       not self._path(e["label"]).is_file()]
            if pending:
                entry = pending[-1]
                if float(entry["amplitude"]) != amplitude:
                    raise ValueError("Resume the pending point before choosing a new one")
                label = entry["label"]
            else:
                label = f"bo_{len(self.observations()):04d}"
            self._acquire(label, amplitude, baseline_id)
            records = self.observations()
            _csv(data_dir(self.run) / "observations.csv", records)
            return records[-1]

    def save_search_state(self, suggestion, rng_state, status="running"):
        with _run_lock(self.run):
            self.manifest = _load_manifest(self.run)
            self.manifest["suggestion"] = suggestion
            self.manifest["rng_state"] = rng_state
            self.manifest["status"] = status
            self._save()
