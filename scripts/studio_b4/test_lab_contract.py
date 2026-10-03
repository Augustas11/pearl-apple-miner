"""B4 lab boundary regressions; uses real B7 ownership checks and hook execution."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

import b4_window as window


def process_group_members(pgid):
    result = subprocess.run(
        ['ps', '-axo', 'pid=,pgid=,args='],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    members = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and int(fields[1]) == pgid:
            members.append((int(fields[0]), fields[2]))
    return members


def wait_until_group_gone(pgid, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_group_members(pgid):
            return True
        time.sleep(0.1)
    return not process_group_members(pgid)


class LabContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.lock = self.home / '.lab-window.lock'
        self.lock.mkdir()
        (self.lock / 'pmk-owner.json').write_text(json.dumps({'token': 'test-owner', 'pause_confirmed': True}))
        self.hook = self.home / 'resume'
        self.report = self.home / 'report.json'
        self.count = self.home / 'calls'
        self.hook.write_text(f'#!/bin/sh\necho called >> "{self.count}"\nprintf \'%s\\n\' \'{{"confirmed":true,"outcome":"resumed"}}\'\n')
        self.hook.chmod(0o700)
        self.env = {
            'B4_LAB_OWNER_TOKEN': 'test-owner',
            'B4_LAB_RESUME_HOOK': str(self.hook),
            'B4_LAB_RESUME_REPORT': str(self.report),
        }

    def test_foreign_lock_cannot_be_adopted_by_reading_its_token(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'home', return_value=self.home):
            with self.assertRaisesRegex(RuntimeError, 'another session'):
                window.prepare_lab_window()
        self.assertFalse(self.count.exists())

    def test_unconfirmed_pause_refused(self):
        (self.lock / 'pmk-owner.json').write_text(json.dumps({'token': 'test-owner', 'pause_confirmed': False}))
        with patch.dict(os.environ, self.env, clear=True), patch.object(Path, 'home', return_value=self.home):
            with self.assertRaisesRegex(RuntimeError, 'pause is not confirmed'):
                window.prepare_lab_window()
        self.assertFalse(self.count.exists())

    def test_token_stays_out_of_argv_and_real_resume_runs_at_controller_finish(self):
        with patch.dict(os.environ, self.env, clear=True), patch.object(Path, 'home', return_value=self.home):
            lab = window.prepare_lab_window()
            args = window.miner_cmd(self.home / 'config.toml', 1234)
            self.assertNotIn('--lab-owner-token', args)
            self.assertNotIn('test-owner', args)
            self.assertEqual(os.environ['PMK_B4_LAB_OWNER_TOKEN'], 'test-owner')
            self.assertEqual(args[args.index('--resume-hook') + 1], str(self.hook))
            self.assertFalse(self.count.exists())
            outcome = window.finish_lab_window(lab)
        self.assertTrue(outcome['confirmed'])
        self.assertEqual(self.count.read_text().splitlines(), ['called'])
        self.assertTrue(json.loads(self.report.read_text())['confirmed'])
        self.assertTrue(self.lock.exists())

    def test_preflight_failure_still_performs_confirmed_resume(self):
        with patch.dict(os.environ, self.env, clear=True), \
             patch.object(Path, 'home', return_value=self.home), \
             patch.object(window, 'SUMMARY', self.home / 'summary.json'), \
             patch.object(window, 'preflight', side_effect=window.StepFailure('test preflight failure')), \
             patch.object(sys, 'argv', ['b4_window.py', '--quick']):
            self.assertEqual(window.main(), 1)
        self.assertEqual(self.count.read_text().splitlines(), ['called'])
        self.assertTrue(json.loads(self.report.read_text())['confirmed'])

    def test_lab_token_cli_flag_rejected(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv', ['b4_window.py', '--lab-owner-token', 'secret']):
            with self.assertRaises(SystemExit) as raised:
                window.main()
        self.assertEqual(raised.exception.code, 2)

    def test_failed_resume_fails_closed_and_records_failure(self):
        self.hook.write_text('#!/bin/sh\nprintf \'%s\\n\' \'{"confirmed":false,"outcome":"failed"}\'\n')
        with patch.dict(os.environ, self.env, clear=True), patch.object(Path, 'home', return_value=self.home):
            lab = window.prepare_lab_window()
            with self.assertRaisesRegex(RuntimeError, 'resume unconfirmed'):
                window.finish_lab_window(lab)
        self.assertFalse(json.loads(self.report.read_text())['confirmed'])

    def test_repeated_cleanup_signals_do_not_skip_resume_or_reaping(self):
        try:
            pearld = window.find_pearld()
        except window.StepFailure as exc:
            self.skipTest(str(exc))

        controller = self.home / 'cleanup_controller.py'
        hook_started = self.home / 'hook_started'
        cleanup_started = self.home / 'cleanup_started'
        ready_path = self.home / 'ready.json'
        queued_path = self.home / 'queued.json'
        result_path = self.home / 'result.json'
        call_log = self.home / 'calls'
        report = self.home / 'report.json'
        summary_path = self.home / 'summary.json'
        datadir = self.home / 'pearld-data'
        logdir = self.home / 'pearld-logs'
        run_root = self.home / 'run'

        self.hook.write_text(
            textwrap.dedent(
                f"""\
                #!/bin/sh
                echo called >> {str(call_log)!r}
                echo started > {str(hook_started)!r}
                sleep 0.5
                printf '%s\\n' '{{"confirmed":true,"outcome":"resumed"}}'
                """
            ),
            encoding='utf-8',
        )
        self.hook.chmod(0o700)

        controller.write_text(
            textwrap.dedent(
                f"""\
                import json
                import os
                import signal
                import sys
                import time
                from pathlib import Path

                sys.path.insert(0, {str(Path(window.__file__).resolve().parent)!r})
                import b4_window as window

                window.RUN_ROOT = Path({str(run_root)!r})
                window.SUMMARY = Path({str(summary_path)!r})
                os.environ['B4_LAB_OWNER_TOKEN'] = 'test-owner'
                os.environ['B4_LAB_RESUME_HOOK'] = {str(self.hook)!r}
                os.environ['B4_LAB_RESUME_REPORT'] = {str(report)!r}

                summary = {{
                    'steps': {{}},
                    'criteria': {{'lab_resume': 'NOT_RUN'}},
                }}
                pgid = None

                def on_interrupt(signum, _frame):
                    raise window.StepFailure(f'interrupted by {{signum}}')

                for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    signal.signal(signum, on_interrupt)

                def start_real_pearld():
                    pearld_proc = window.start_proc(
                        'pearld',
                        [
                            str(Path({str(pearld)!r})),
                            '--regtest',
                            '--nodnsseed',
                            '--nolisten',
                            '--norpc',
                            '--connect=127.0.0.1:1',
                            {f'--datadir={datadir}'!r},
                            {f'--logdir={logdir}'!r},
                        ],
                        Path({str(self.home / 'pearld.log')!r}),
                        env=window.bundle_env(),
                    )
                    return pearld_proc.process.pid

                def observe_cleanup(frame, event, arg):
                    # Observe real cleanup without replacing any lifecycle code.
                    if frame.f_code is window.stop_all.__code__ and event == 'line':
                        expected = {{signal.SIGTERM, signal.SIGHUP}}
                        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
                        if expected <= blocked:
                            Path({str(cleanup_started)!r}).touch()
                            pending = signal.sigpending()
                            if expected <= pending:
                                Path({str(queued_path)!r}).write_text(json.dumps(sorted(map(int, pending))))
                    return observe_cleanup

                lab = window.prepare_lab_window()
                sys.settrace(observe_cleanup)
                try:
                    pgid = start_real_pearld()
                    Path({str(ready_path)!r}).write_text(json.dumps({{'pgid': pgid, 'pid': os.getpid()}}))
                    while True:
                        time.sleep(0.1)
                except window.StepFailure as exc:
                    summary['error'] = str(exc)
                finally:
                    window.cleanup_lab_window(lab, summary)
                Path({str(result_path)!r}).write_text(json.dumps({{'summary': summary, 'pgids': [pgid] if pgid else []}}))
                """
            ),
            encoding='utf-8',
        )

        env = os.environ.copy()
        env['HOME'] = str(self.home)
        env['PYTHONPATH'] = os.pathsep.join(
            [str(Path(window.__file__).resolve().parent), str(window.ROOT / 'miner'), env.get('PYTHONPATH', '')]
        ).rstrip(os.pathsep)
        process = subprocess.Popen(
            [sys.executable, str(controller)],
            cwd=window.ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        pgid = None
        output = ''
        cleanup_signals_sent = False
        hook_signals_sent = False
        try:
            deadline = time.monotonic() + 15
            while not ready_path.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if not ready_path.exists() and process.poll() is not None:
                output += process.communicate(timeout=5)[0]
            self.assertTrue(ready_path.exists(), output)
            ready = json.loads(ready_path.read_text(encoding='utf-8'))
            pgid = int(ready['pgid'])
            deadline = time.monotonic() + 10
            while (
                not any(
                    pid != pgid and 'pearld' in command and 'process_guard.py' not in command
                    for pid, command in process_group_members(pgid)
                )
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertTrue(
                any(
                    pid != pgid and 'pearld' in command and 'process_guard.py' not in command
                    for pid, command in process_group_members(pgid)
                ),
                process_group_members(pgid),
            )
            os.killpg(pgid, signal.SIGSTOP)
            os.kill(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while not cleanup_started.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            self.assertTrue(cleanup_started.exists(), output)
            self.assertIsNone(process.poll(), output)
            os.kill(process.pid, signal.SIGTERM)
            os.kill(process.pid, signal.SIGHUP)
            cleanup_signals_sent = True
            deadline = time.monotonic() + 15
            while not hook_started.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if not hook_started.exists() and process.poll() is not None:
                output += process.communicate(timeout=5)[0]
            self.assertTrue(hook_started.exists(), output)
            os.kill(process.pid, signal.SIGHUP)
            os.kill(process.pid, signal.SIGTERM)
            hook_signals_sent = True
            stdout, _ = process.communicate(timeout=30)
            output += stdout
        finally:
            if process.poll() is None:
                process.kill()
                output += process.communicate(timeout=5)[0]
            if pgid is not None and process_group_members(pgid):
                os.killpg(pgid, signal.SIGKILL)
                wait_until_group_gone(pgid)
        self.assertTrue(queued_path.exists(), 'TERM and HUP were not both queued during stop_all')
        self.assertTrue({int(signal.SIGTERM), int(signal.SIGHUP)} <= set(json.loads(queued_path.read_text())))
        self.assertTrue(cleanup_signals_sent)
        self.assertTrue(hook_signals_sent)
        self.assertEqual(process.returncode, 0, output)
        result = json.loads(result_path.read_text(encoding='utf-8'))
        self.assertEqual(call_log.read_text(encoding='utf-8').splitlines(), ['called'])
        self.assertEqual(result['summary']['criteria']['lab_resume'], 'PASS')
        self.assertEqual(result['summary']['steps']['lab_resume']['outcome'], 'resumed')
        self.assertTrue(json.loads(report.read_text(encoding='utf-8'))['confirmed'])
        for pgid in result['pgids']:
            self.assertTrue(wait_until_group_gone(int(pgid)), process_group_members(int(pgid)))


if __name__ == '__main__':
    unittest.main()
