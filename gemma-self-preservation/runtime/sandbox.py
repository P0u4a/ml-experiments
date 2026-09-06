from __future__ import annotations

import os
import pwd
import shutil
import subprocess
from typing import Any


def lock_down_agent_network(user: str = "dev") -> dict[str, Any]:
    """Deny all IPv4/IPv6 egress from agent shell processes by UID.

    The trusted root controller retains networking so it can call the host
    llama.cpp server. Agent tools are always privilege-dropped to ``dev`` and
    therefore match these kernel OUTPUT rules, including commands launched by
    package scripts or code written during the episode.
    """

    if os.geteuid() != 0:
        raise RuntimeError("agent network lockdown must be installed by root")
    uid = pwd.getpwnam(user).pw_uid
    installed: list[dict[str, Any]] = []
    for family, executable in (("ipv4", "iptables"), ("ipv6", "ip6tables")):
        binary = shutil.which(executable)
        if binary is None:
            raise RuntimeError(f"required network sandbox binary is missing: {executable}")
        rule = ["OUTPUT", "-m", "owner", "--uid-owner", str(uid), "-j", "REJECT"]
        check = subprocess.run(
            [binary, "--wait", "5", "-C", *rule],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if check.returncode != 0:
            add = subprocess.run(
                [binary, "--wait", "5", "-I", *rule],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if add.returncode != 0:
                raise RuntimeError(
                    f"could not install {family} agent egress rule: {add.stdout.strip()}"
                )
        verify = subprocess.run(
            [binary, "--wait", "5", "-C", *rule],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if verify.returncode != 0:
            raise RuntimeError(
                f"could not verify {family} agent egress rule: {verify.stdout.strip()}"
            )
        installed.append({"family": family, "uid": uid, "target": "REJECT"})
    return {"user": user, "uid": uid, "rules": installed}
