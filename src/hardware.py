"""LQCloud job reconstruction is isolated from calibration logic."""


def restore_job(backend, job_id, use_async=None):
    from lqcloud.job.job import Job
    async_flag = bool(getattr(backend, "use_async", True)) if use_async is None else bool(use_async)
    return Job(job_id, backend.provider, backend.name,
               base_url=backend.base_url, use_async=async_flag)
