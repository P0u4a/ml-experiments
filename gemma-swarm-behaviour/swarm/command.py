from __future__ import annotations

import os
import pwd
import subprocess


AGENT_USER = "dev"


def _agent_account() -> pwd.struct_passwd | None:
    try:
        return pwd.getpwnam(AGENT_USER)
    except KeyError:
        return None


def run_command(command: str, *, timeout: int = 30, cwd: str = "/agent") -> tuple[int, str]:
    """Run an agent shell command with dropped privileges and no harness secrets."""
    account = _agent_account()
    if os.geteuid() == 0 and account is None:
        raise RuntimeError(f"container is missing the required {AGENT_USER!r} user")

    environment = None
    preexec = None
    if account is not None:
        environment = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": account.pw_dir,
            "USER": AGENT_USER,
            "LOGNAME": AGENT_USER,
            "SHELL": "/bin/bash",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TERM": os.environ.get("TERM", "xterm"),
        }
        if os.geteuid() == 0:
            def drop_privileges() -> None:
                os.setgid(account.pw_gid)
                os.initgroups(AGENT_USER, account.pw_gid)
                os.setuid(account.pw_uid)

            preexec = drop_privileges

    run_cwd = cwd if os.path.isdir(cwd) else None
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            close_fds=True,
            timeout=timeout,
            cwd=run_cwd,
            preexec_fn=preexec,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return 75, f"<TIMEOUT>{command}</TIMEOUT>"
    return result.returncode, result.stdout.decode(errors="replace")
