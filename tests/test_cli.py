"""Behavior tests for the production daily-analysis CLI."""

from __future__ import annotations

import asyncio
import time
import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stockbot import cli
from stockbot.screening.universe import Ticker


VALID_ENV = {
    "OPENAI_API_KEY": "test-openai",
    "FINANCIAL_DATASETS_API_KEY": "test-financial-datasets",
}
FIXED_NOW = datetime(2026, 8, 31, 12, 34, 56, tzinfo=timezone.utc)
TEST_UNIVERSE = [
    Ticker("AAPL", "US", "sp500"),
    Ticker("SMALL", "US", "sp600"),
    Ticker("SHOP.TO", "TSX", "tsx_composite"),
]


class FakeFlow:
    def __init__(self, report: str = "# Complete report\n\nAll sections.\n") -> None:
        self.report = report

        self.universes: list[list[Ticker] | None] = []

    async def execute_undervalued_analysis(
        self, universe: list[Ticker] | None = None
    ) -> str:
        self.universes.append(universe)
        return self.report


def invoke_cli(
    tmp_path: Path,
    args: list[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
    flow_factory=None,
    universe_loader=None,
) -> tuple[int, list[object]]:
    captured_preferences: list[object] = []

    def default_factory(preferences):
        captured_preferences.append(preferences)
        return FakeFlow()

    exit_code = cli.main(
        ["daily", "--output-dir", str(tmp_path / "reports"), *(args or [])],
        environ=VALID_ENV if environ is None else environ,
        flow_factory=flow_factory or default_factory,
        state_dir=tmp_path / "state",
        now=lambda: FIXED_NOW,
        load_environment=lambda: None,
        universe_loader=universe_loader or (lambda: TEST_UNIVERSE),
    )
    return exit_code, captured_preferences


def result_payload(stdout: str) -> dict[str, object]:
    marker_lines = [
        line for line in stdout.splitlines() if line.startswith("STOCKBOT_RESULT_JSON=")
    ]
    assert len(marker_lines) == 1
    return json.loads(marker_lines[0].split("=", 1)[1])


def test_daily_defaults_match_discord_preferences(tmp_path, capsys):
    exit_code, captured = invoke_cli(tmp_path)

    assert exit_code == cli.EXIT_OK
    assert len(captured) == 1
    assert captured[0].model_dump() == {
        "max_price": 100.0,
        "min_price": 5.0,
        "min_volume": 500000.0,
        "max_pe": 25.0,
        "min_market_cap": 300000000.0,
        "min_current_ratio": 1.5,
        "max_debt_equity": 2.0,
        "price_vs_high": 0.4,
    }
    assert capsys.readouterr().err == ""


def test_daily_default_filters_to_sp500_and_loads_universe_once(tmp_path):
    flow = FakeFlow()
    loader_calls = 0

    def loader():
        nonlocal loader_calls
        loader_calls += 1
        return TEST_UNIVERSE

    exit_code, _ = invoke_cli(
        tmp_path,
        flow_factory=lambda _preferences: flow,
        universe_loader=loader,
    )

    assert exit_code == cli.EXIT_OK
    assert loader_calls == 1
    assert flow.universes == [[TEST_UNIVERSE[0]]]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("sp500", [TEST_UNIVERSE[0]]),
        ("sp600", [TEST_UNIVERSE[1]]),
        ("tsx", [TEST_UNIVERSE[2]]),
    ],
)
def test_explicit_source_choices_filter_by_ticker_source(
    tmp_path, choice, expected
):
    flow = FakeFlow()

    exit_code, _ = invoke_cli(
        tmp_path,
        ["--universe", choice],
        flow_factory=lambda _preferences: flow,
    )

    assert exit_code == cli.EXIT_OK
    assert flow.universes == [expected]


def test_full_universe_is_explicit_opt_in(tmp_path):
    flow = FakeFlow()

    exit_code, _ = invoke_cli(
        tmp_path,
        ["--universe", "full"],
        flow_factory=lambda _preferences: flow,
    )

    assert exit_code == cli.EXIT_OK
    assert flow.universes == [[TEST_UNIVERSE[0], TEST_UNIVERSE[2], TEST_UNIVERSE[1]]]


def test_rotating_window_is_sorted_unique_and_deterministic():
    universe = [
        Ticker("DDD", "US", "sp500"),
        Ticker("AAA", "US", "sp500"),
        Ticker("CCC", "US", "sp500"),
        Ticker("AAA", "US", "sp500"),
        Ticker("BBB", "US", "sp500"),
    ]

    first = cli._rotating_symbol_window(universe, 3, FIXED_NOW.date())
    repeated = cli._rotating_symbol_window(universe, 3, FIXED_NOW.date())
    following = cli._rotating_symbol_window(
        universe, 3, FIXED_NOW.date().fromordinal(FIXED_NOW.date().toordinal() + 1)
    )

    assert first == repeated
    assert len(first) == len({ticker.symbol for ticker in first}) == 3
    assert [ticker.symbol for ticker in first] != [ticker.symbol for ticker in following]


def test_rotating_windows_cover_source_over_successive_days():
    universe = [Ticker(f"S{i:03d}", "US", "sp500") for i in range(503)]
    seen = set()

    for offset in range(2):
        selection_date = FIXED_NOW.date().fromordinal(
            FIXED_NOW.date().toordinal() + offset
        )
        seen.update(
            ticker.symbol
            for ticker in cli._rotating_symbol_window(universe, 300, selection_date)
        )

    assert seen == {ticker.symbol for ticker in universe}


@pytest.mark.parametrize("cap", [0, 3, 99])
def test_zero_or_large_cap_selects_full_unique_source(cap):
    universe = [*TEST_UNIVERSE, TEST_UNIVERSE[0]]

    selected = cli._rotating_symbol_window(universe, cap, FIXED_NOW.date())

    assert [ticker.symbol for ticker in selected] == ["AAPL", "SHOP.TO", "SMALL"]


def test_daily_default_caps_source_at_300(tmp_path):
    flow = FakeFlow()
    universe = [Ticker(f"S{i:03d}", "US", "sp500") for i in range(503)]

    exit_code, _ = invoke_cli(
        tmp_path,
        flow_factory=lambda _preferences: flow,
        universe_loader=lambda: universe,
    )

    assert exit_code == cli.EXIT_OK
    assert len(flow.universes[0]) == 300


def test_max_symbols_zero_opts_into_full_selected_source(tmp_path):
    flow = FakeFlow()
    universe = [Ticker(f"S{i:03d}", "US", "sp500") for i in range(503)]

    exit_code, _ = invoke_cli(
        tmp_path,
        ["--max-symbols", "0"],
        flow_factory=lambda _preferences: flow,
        universe_loader=lambda: universe,
    )

    assert exit_code == cli.EXIT_OK
    assert flow.universes == [universe]


def test_negative_max_symbols_fails_before_configuration_or_universe_work(
    tmp_path, capsys
):
    loader_called = False

    def loader():
        nonlocal loader_called
        loader_called = True
        return TEST_UNIVERSE

    exit_code, _ = invoke_cli(
        tmp_path,
        ["--max-symbols", "-1"],
        environ={},
        universe_loader=loader,
    )

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE
    assert loader_called is False
    assert "max symbols must be at least 0" in output.err


def test_empty_universe_selection_fails_without_calling_flow(tmp_path, capsys):
    factory_called = False

    def factory(_preferences):
        nonlocal factory_called
        factory_called = True
        return FakeFlow()

    exit_code, _ = invoke_cli(
        tmp_path,
        ["--universe", "tsx"],
        flow_factory=factory,
        universe_loader=lambda: TEST_UNIVERSE[:2],
    )

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert factory_called is False
    assert "selected universe 'tsx' is empty" in output.err
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not (tmp_path / "reports").exists()


def test_daily_preference_flags_reach_flow(tmp_path):
    args = [
        "--min-price", "2.5", "--max-price", "75", "--min-volume", "120000",
        "--max-pe", "18", "--min-market-cap", "500000000",
        "--min-current-ratio", "1.8", "--max-debt-equity", "1.2",
        "--max-decline-from-high", "0.25",
    ]

    exit_code, captured = invoke_cli(tmp_path, args)

    assert exit_code == cli.EXIT_OK
    prefs = captured[0]
    assert prefs.min_price == 2.5
    assert prefs.max_price == 75.0
    assert prefs.min_volume == 120000.0
    assert prefs.max_pe == 18.0
    assert prefs.min_market_cap == 500000000.0
    assert prefs.min_current_ratio == 1.8
    assert prefs.max_debt_equity == 1.2
    assert prefs.price_vs_high == 0.25


@pytest.mark.parametrize(
    "args",
    [
        ["--min-price", "-1"],
        ["--min-price", "20", "--max-price", "10"],
        ["--min-volume", "-1"],
        ["--max-pe", "0"],
        ["--min-market-cap", "-1"],
        ["--min-current-ratio", "-0.1"],
        ["--max-debt-equity", "-0.1"],
        ["--max-decline-from-high", "1.01"],
    ],
)
def test_invalid_preferences_exit_before_flow(tmp_path, capsys, args):
    called = False

    def factory(_preferences):
        nonlocal called
        called = True
        return FakeFlow()

    exit_code, _ = invoke_cli(tmp_path, args, flow_factory=factory)

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE
    assert called is False
    assert "error:" in captured.err.lower()
    assert "STOCKBOT_RESULT_JSON=" not in captured.out
    assert not (tmp_path / "reports").exists()


def test_output_file_and_output_dir_are_mutually_exclusive(tmp_path, capsys):
    exit_code = cli.main(
        ["daily", "--output-dir", str(tmp_path / "one"), "--output-file", str(tmp_path / "two.md")],
        environ=VALID_ENV,
        flow_factory=lambda _preferences: FakeFlow(),
        state_dir=tmp_path / "state",
        load_environment=lambda: None,
    )

    assert exit_code == cli.EXIT_USAGE
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("missing_key", sorted(VALID_ENV))
def test_missing_required_key_creates_no_report(tmp_path, capsys, missing_key):
    environ = {key: value for key, value in VALID_ENV.items() if key != missing_key}

    exit_code, captured = invoke_cli(tmp_path, environ=environ)

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_CONFIG
    assert captured == []
    assert missing_key in output.err
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not (tmp_path / "reports").exists()


def test_success_saves_complete_report_atomically(tmp_path, capsys, monkeypatch):
    report = "# Daily\n\n" + ("complete content\n" * 100)
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = cli.os.replace

    def recording_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", recording_replace)
    exit_code, _ = invoke_cli(
        tmp_path, flow_factory=lambda _preferences: FakeFlow(report)
    )

    output = capsys.readouterr()
    payload = result_payload(output.out)
    report_path = Path(str(payload["report_path"]))
    assert exit_code == cli.EXIT_OK
    assert report_path.read_text(encoding="utf-8") == report
    assert payload["report_bytes"] == len(report.encode("utf-8"))
    assert replace_calls and replace_calls[-1][1] == report_path
    assert not list(report_path.parent.glob("*.tmp"))


def test_explicit_output_file_is_exact_and_absolute_in_marker(tmp_path, capsys):
    requested = tmp_path / "automation" / "morning.md"
    exit_code = cli.main(
        ["daily", "--output-file", str(requested)],
        environ=VALID_ENV,
        flow_factory=lambda _preferences: FakeFlow("exact report"),
        state_dir=tmp_path / "state",
        now=lambda: FIXED_NOW,
        load_environment=lambda: None,
        universe_loader=lambda: TEST_UNIVERSE,
    )

    payload = result_payload(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert requested.read_text(encoding="utf-8") == "exact report"
    assert payload["report_path"] == str(requested.resolve())


def test_default_filename_is_timestamped_and_collision_safe(tmp_path, capsys):
    first_code, _ = invoke_cli(tmp_path)
    first_payload = result_payload(capsys.readouterr().out)
    second_code, _ = invoke_cli(tmp_path)
    second_payload = result_payload(capsys.readouterr().out)

    assert first_code == second_code == cli.EXIT_OK
    assert Path(str(first_payload["report_path"])).name == "daily_undervalued_20260831T123456Z.md"
    assert Path(str(second_payload["report_path"])).name == "daily_undervalued_20260831T123456Z_1.md"
    assert first_payload["report_path"] != second_payload["report_path"]


def test_lock_contention_has_distinct_exit_code(tmp_path, capsys):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lock_path = state_dir / cli.LOCK_FILENAME
    with lock_path.open("a+") as held_lock:
        fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        exit_code, captured = invoke_cli(tmp_path)

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_LOCKED
    assert captured == []
    assert "already running" in output.err.lower()
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not (tmp_path / "reports").exists()


def test_flow_exception_leaves_no_report_or_partial_file(tmp_path, capsys):
    class BrokenFlow:
        async def execute_undervalued_analysis(self, universe=None):
            raise RuntimeError("provider unavailable")

    exit_code, _ = invoke_cli(
        tmp_path, flow_factory=lambda _preferences: BrokenFlow()
    )

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert "provider unavailable" in output.err
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not list((tmp_path / "reports").glob("*"))


def test_scan_quality_failure_has_no_success_marker_or_partial_report(tmp_path, capsys):
    class QualityFailureFlow:
        async def execute_undervalued_analysis(self, universe=None):
            raise RuntimeError(
                "numeric scan quality below threshold: 640 failed of 1323 (48.4% > 20.0%)"
            )

    exit_code, _ = invoke_cli(
        tmp_path, flow_factory=lambda _preferences: QualityFailureFlow()
    )

    output = capsys.readouterr()
    assert exit_code == 1
    assert "640 failed of 1323" in output.err
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not (tmp_path / "reports").exists()


def test_lock_is_released_after_failure(tmp_path):
    class BrokenFlow:
        async def execute_undervalued_analysis(self, universe=None):
            raise RuntimeError("first run failed")

    first_code, _ = invoke_cli(
        tmp_path, flow_factory=lambda _preferences: BrokenFlow()
    )
    second_code, _ = invoke_cli(tmp_path)

    assert first_code == cli.EXIT_RUNTIME
    assert second_code == cli.EXIT_OK


def test_success_marker_schema_and_exit_code(tmp_path, capsys):
    exit_code, _ = invoke_cli(tmp_path)

    payload = result_payload(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert payload == {
        "status": "ok",
        "report_path": payload["report_path"],
        "run_type": "daily_undervalued",
        "generated_at": "2026-08-31T12:34:56Z",
        "report_bytes": len(FakeFlow().report.encode("utf-8")),
        "universe": "sp500",
        "universe_size": 1,
        "max_symbols": 300,
        "source_universe_size": 1,
        "selected_universe_size": 1,
        "selection_date": "2026-08-31",
    }
    assert Path(str(payload["report_path"])).is_absolute()


def test_injected_now_is_shared_by_selection_and_generated_at(tmp_path, capsys):
    calls = 0

    def ticking_now():
        nonlocal calls
        calls += 1
        if calls == 1:
            return FIXED_NOW
        return datetime(2026, 9, 1, tzinfo=timezone.utc)

    exit_code = cli.main(
        ["daily", "--output-dir", str(tmp_path / "reports")],
        environ=VALID_ENV,
        flow_factory=lambda _preferences: FakeFlow(),
        state_dir=tmp_path / "state",
        now=ticking_now,
        load_environment=lambda: None,
        universe_loader=lambda: TEST_UNIVERSE,
    )

    payload = result_payload(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert calls == 1
    assert payload["generated_at"] == "2026-08-31T12:34:56Z"
    assert payload["selection_date"] == "2026-08-31"


def test_default_state_dir_is_canonical_and_independent_of_cwd(tmp_path):
    expected = Path(cli.__file__).resolve().parent.parent / "state"
    assert cli.DEFAULT_STATE_DIR == expected
    assert cli.DEFAULT_STATE_DIR.is_absolute()

    lock_path = cli.DEFAULT_STATE_DIR / cli.LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as held_lock:
        fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = "from stockbot import cli; raise SystemExit(cli.main(['daily']))"
        environment = {
            **os.environ,
            **VALID_ENV,
            "PYTHONPATH": str(Path(cli.__file__).resolve().parent.parent),
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    assert result.returncode == cli.EXIT_LOCKED
    assert "already running" in result.stderr.lower()
    assert "STOCKBOT_RESULT_JSON=" not in result.stdout


@pytest.mark.parametrize("relative_target", ["daily_analysis.lock", "reports/out.md"])
def test_output_file_inside_state_directory_is_rejected(
    tmp_path, capsys, relative_target
):
    state_dir = tmp_path / "state"
    target = state_dir / relative_target

    exit_code = cli.main(
        ["daily", "--output-file", str(target)],
        environ=VALID_ENV,
        flow_factory=lambda _preferences: FakeFlow(),
        state_dir=state_dir,
        now=lambda: FIXED_NOW,
        load_environment=lambda: None,
        universe_loader=lambda: TEST_UNIVERSE,
    )

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert "protected state directory" in output.err
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not target.exists() or target.name == cli.LOCK_FILENAME


def test_generated_output_directory_inside_state_is_rejected(tmp_path, capsys):
    state_dir = tmp_path / "state"
    exit_code = cli.main(
        ["daily", "--output-dir", str(state_dir / "reports")],
        environ=VALID_ENV,
        flow_factory=lambda _preferences: FakeFlow(),
        state_dir=state_dir,
        now=lambda: FIXED_NOW,
        load_environment=lambda: None,
        universe_loader=lambda: TEST_UNIVERSE,
    )

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert "protected state directory" in output.err
    assert not (state_dir / "reports").exists()


def _reserve_path_worker(output_dir: str, start, results) -> None:
    start.wait()
    path = cli._reserve_collision_safe_path(Path(output_dir), "20260831T123456Z")
    results.put(str(path))


def test_generated_filename_reservation_is_atomic_across_processes(tmp_path):
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    workers = [
        context.Process(target=_reserve_path_worker, args=(str(output_dir), start, results))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    start.set()
    paths = [Path(results.get(timeout=10)) for _ in workers]
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0

    assert len(set(paths)) == 2
    assert all(path.exists() for path in paths)


def test_load_environment_oserror_is_concise_runtime_failure(tmp_path, capsys):
    def broken_loader():
        raise OSError("dotenv unreadable")

    exit_code = cli.main(
        ["daily", "--output-dir", str(tmp_path / "reports")],
        environ=VALID_ENV,
        load_environment=broken_loader,
    )

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert output.err.strip() == "error: environment loading failed: dotenv unreadable"
    assert output.out == ""


@pytest.mark.parametrize("timeout", ["0", "-1", "inf", "nan"])
def test_timeout_must_be_finite_and_positive(tmp_path, capsys, timeout):
    exit_code, captured = invoke_cli(tmp_path, ["--timeout", timeout])
    output = capsys.readouterr()
    assert exit_code == cli.EXIT_USAGE
    assert captured == []
    assert "timeout must be finite and greater than 0" in output.err


def test_flow_timeout_leaves_no_report_marker_or_reservation(tmp_path, capsys):
    class SlowFlow:
        async def execute_undervalued_analysis(self, universe=None):
            await asyncio.sleep(1)
            return "too late"

    exit_code, _ = invoke_cli(
        tmp_path,
        ["--timeout", "0.01"],
        flow_factory=lambda _preferences: SlowFlow(),
    )

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert "timed out after 0.01 seconds" in output.err
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not list((tmp_path / "reports").glob("*"))


def test_supervised_worker_bounds_blocking_synchronous_flow_and_exits_nonzero(
    tmp_path, capsys
):
    class BlockingFlow:
        async def execute_undervalued_analysis(self, universe=None):
            time.sleep(5)
            return "too late"

    started = time.monotonic()
    exit_code = cli.main(
        ["daily", "--output-dir", str(tmp_path / "reports"), "--timeout", "0.1"],
        environ=VALID_ENV,
        flow_factory=lambda _preferences: BlockingFlow(),
        supervise_flow=True,
        state_dir=tmp_path / "state",
        now=lambda: FIXED_NOW,
        load_environment=lambda: None,
        universe_loader=lambda: TEST_UNIVERSE,
    )

    assert time.monotonic() - started < 1.5
    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert "timed out after 0.1 seconds" in output.err
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not list((tmp_path / "reports").glob("*"))


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_generated_reservation_and_temporary_are_cleaned_on_publish_failure(
    tmp_path, capsys, monkeypatch, failure_point
):
    if failure_point == "write":
        def fail_write(*_args):
            raise OSError("write failed")

        monkeypatch.setattr(cli, "_write_temporary_report", fail_write)
    else:
        def fail_replace(*_args):
            raise OSError("replace failed")

        monkeypatch.setattr(cli.os, "replace", fail_replace)

    exit_code, _ = invoke_cli(tmp_path)

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert f"{failure_point} failed" in output.err
    assert "STOCKBOT_RESULT_JSON=" not in output.out
    assert not list((tmp_path / "reports").glob("*"))


def test_atomic_publish_fsyncs_parent_directory(tmp_path, monkeypatch):
    target = tmp_path / "reports" / "report.md"
    synced: list[Path] = []
    monkeypatch.setattr(cli, "_fsync_directory", lambda path: synced.append(path))

    cli._atomic_write(target, "durable")

    assert target.read_text(encoding="utf-8") == "durable"
    assert synced == [target.parent]


def test_whitespace_only_report_is_rejected(tmp_path, capsys):
    exit_code, _ = invoke_cli(
        tmp_path, flow_factory=lambda _preferences: FakeFlow(" \n\t")
    )

    output = capsys.readouterr()
    assert exit_code == cli.EXIT_RUNTIME
    assert "empty report" in output.err
    assert not list((tmp_path / "reports").glob("*"))


def test_explicit_automation_output_still_replaces_existing_file(tmp_path, capsys):
    target = tmp_path / "automation" / "latest.md"
    target.parent.mkdir()
    target.write_text("old", encoding="utf-8")

    exit_code = cli.main(
        ["daily", "--output-file", str(target), "--timeout", "2"],
        environ=VALID_ENV,
        flow_factory=lambda _preferences: FakeFlow("new"),
        state_dir=tmp_path / "state",
        now=lambda: FIXED_NOW,
        load_environment=lambda: None,
        universe_loader=lambda: TEST_UNIVERSE,
    )

    assert exit_code == cli.EXIT_OK
    assert target.read_text(encoding="utf-8") == "new"
    assert result_payload(capsys.readouterr().out)["report_path"] == str(target.resolve())
