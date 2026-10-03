#!/usr/bin/env python3
"""Run one command in a controller-owned process group.

macOS has no Linux ``PR_SET_PDEATHSIG``.  The controller therefore keeps the
write end of a pipe open and passes only the read end to this guard.  If the
controller exits for any reason, including SIGKILL, EOF becomes readable and
the guard kills its complete process group before any child can be orphaned.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
import time


def group_members() -> list[int]:
    """Return the current group's PIDs except this guard."""
    probe = subprocess.Popen(
        ["ps", "-axo", "pid=,pgid="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        close_fds=True,
    )
    stdout, _ = probe.communicate()
    group = os.getpgrp()
    own_pid = os.getpid()
    members: list[int] = []
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and int(fields[1]) == group and int(fields[0]) not in {own_pid, probe.pid}:
            members.append(int(fields[0]))
    return members


def reap_residual_group() -> None:
    """Kill descendants left behind after the direct payload exits."""
    members = group_members()
    for pid in members:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 1.0
    while members and time.monotonic() < deadline:
        time.sleep(0.05)
        members = group_members()
    for pid in members:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("usage: process_guard.py SENTINEL_FD -- COMMAND [ARG ...]")
    sentinel_fd = int(sys.argv[1])
    command = sys.argv[3:]
    os.set_blocking(sentinel_fd, False)
    guarded_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        guarded_signals.append(signal.SIGHUP)
    # The controller blocks these around spawn/ownership registration.  Do not
    # propagate that mask into the payload process.
    for signum in guarded_signals:
        # A Python handler is reset to SIG_DFL across exec, so the payload gets
        # ordinary signal behavior while the guard remains alive to finish the
        # TERM/wait/KILL contract or observe controller EOF.
        signal.signal(signum, lambda _signum, _frame: None)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, guarded_signals)
    child = subprocess.Popen(command, close_fds=True)
    selector = selectors.DefaultSelector()
    selector.register(sentinel_fd, selectors.EVENT_READ)
    try:
        while child.poll() is None:
            if selector.select(timeout=0.1):
                try:
                    value = os.read(sentinel_fd, 1)
                except BlockingIOError:
                    continue
                if not value:
                    # The guard is the process-group leader.  SIGKILL is
                    # intentional here: controller death leaves nobody able to
                    # perform a graceful TERM/wait/KILL sequence.
                    os.killpg(os.getpgrp(), signal.SIGKILL)
        returncode = int(child.returncode or 0)
        reap_residual_group()
        if returncode < 0:
            # Match direct Popen semantics for callers that classify signals.
            if -returncode not in (signal.SIGKILL, signal.SIGSTOP):
                signal.signal(-returncode, signal.SIG_DFL)
            os.kill(os.getpid(), -returncode)
        return returncode
    finally:
        selector.close()
        os.close(sentinel_fd)


if __name__ == "__main__":
    raise SystemExit(main())
