"""Drive one Rust test in a PTY, typing exit after each Workflow outcome."""
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time

with tempfile.TemporaryDirectory(prefix="workflow-dashboard-test-") as temporary:
    control = Path(temporary) / "control.json"
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 160, 0, 0))
    before = termios.tcgetattr(slave)
    environment = dict(os.environ, TERM="xterm-256color",
                       WORKFLOW_DASHBOARD_TEST=sys.argv[2],
                       WORKFLOW_DASHBOARD_CONTROL=str(control))
    child = subprocess.Popen([sys.argv[1], "--exact", sys.argv[2], "--nocapture"],
                             stdin=slave, stdout=slave, stderr=slave, env=environment)
    output = bytearray()
    exited = set()
    try:
        deadline = time.monotonic() + 120
        while child.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.02)[0]:
                try:
                    output.extend(os.read(master, 65536))
                    del output[:-131072]
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
            if control.is_file():
                document = json.loads(control.read_text())
                previous = {Path(path) for path in document["existing"]}
                for execution in set(Path(document["root"]).glob("execution-*")) - previous:
                    log = execution / "log.txt"
                    if execution in exited or not log.is_file():
                        continue
                    text = log.read_text()
                    if any(marker in text for marker in (
                        "workflow: completed", "workflow: failed", "workflow: cancelled"
                    )):
                        os.write(master, b"exit\r")
                        exited.add(execution)
        assert child.poll() is not None, "dashboard test timed out"
        assert child.returncode == 0, output.decode(errors="replace")
        assert exited, "selected test did not execute a dashboard"
        assert termios.tcgetattr(slave) == before, "terminal state was not restored"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        os.close(master)
        os.close(slave)
