"""Stable command-line entrypoints for unattended stock analysis runs."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
import multiprocessing
import os
import signal
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Mapping, Protocol, Sequence, TextIO

from dotenv import load_dotenv

from stockbot.screening.universe import Ticker

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_LOCKED = 4

LOCK_FILENAME = "daily_analysis.lock"
RUN_TYPE = "daily_undervalued"
DEFAULT_OUTPUT_DIR = Path("reports")
APPLICATION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_DIR = APPLICATION_ROOT / "state"
DEFAULT_TIMEOUT = 900.0
REQUIRED_ENVIRONMENT = ("OPENAI_API_KEY", "FINANCIAL_DATASETS_API_KEY")
UNIVERSE_CHOICES = ("sp500", "sp600", "tsx", "full")
UNIVERSE_SOURCES = {
    "sp500": "sp500",
    "sp600": "sp600",
    "tsx": "tsx_composite",
}


class CliUsageError(ValueError):
    """Raised when command-line arguments fail parsing or validation."""


class LockBusyError(RuntimeError):
    """Raised when another daily analysis holds the process lock."""


class AnalysisFlow(Protocol):
    async def execute_undervalued_analysis(
        self, universe: list[Ticker] | None = None
    ) -> str:
        """Run the analysis and return the complete Markdown report."""


class RaisingArgumentParser(argparse.ArgumentParser):
    """Argument parser that returns errors to the CLI boundary."""

    def error(self, message: str) -> None:
        raise CliUsageError(message)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI argument parser."""
    parser = RaisingArgumentParser(prog="python -m stockbot.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    daily = commands.add_parser(
        "daily",
        help="run the deterministic-first undervalued analysis",
    )
    output = daily.add_mutually_exclusive_group()
    output.add_argument(
        "--output-dir",
        type=Path,
        help=f"report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    output.add_argument(
        "--output-file",
        type=Path,
        help="exact report path for automation",
    )
    daily.add_argument("--min-price", type=float, default=5.0)
    daily.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"overall flow timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    daily.add_argument(
        "--max-symbols",
        type=int,
        default=300,
        help="daily rotating symbol cap (default: 300; 0 disables the cap)",
    )
    daily.add_argument(
        "--universe",
        choices=UNIVERSE_CHOICES,
        default="sp500",
        help="stock universe (default: sp500; full is explicit opt-in)",
    )
    daily.add_argument("--max-price", type=float, default=100.0)
    daily.add_argument("--min-volume", type=float, default=500000.0)
    daily.add_argument("--max-pe", type=float, default=25.0)
    daily.add_argument("--min-market-cap", type=float, default=300000000.0)
    daily.add_argument("--min-current-ratio", type=float, default=1.5)
    daily.add_argument("--max-debt-equity", type=float, default=2.0)
    daily.add_argument(
        "--max-decline-from-high",
        type=float,
        default=0.4,
        help="maximum 52-week-high decline as a fraction from 0 to 1",
    )
    return parser


def _validate_daily_arguments(args: argparse.Namespace) -> None:
    values = {
        "min price": args.min_price,
        "max price": args.max_price,
        "min volume": args.min_volume,
        "max P/E": args.max_pe,
        "min market cap": args.min_market_cap,
        "min current ratio": args.min_current_ratio,
        "max debt/equity": args.max_debt_equity,
        "max decline from high": args.max_decline_from_high,
    }
    for label, value in values.items():
        if not math.isfinite(value):
            raise CliUsageError(f"{label} must be finite")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise CliUsageError("timeout must be finite and greater than 0")
    if args.min_price < 0:
        raise CliUsageError("min price must be at least 0")
    if args.max_symbols < 0:
        raise CliUsageError("max symbols must be at least 0")
    if args.max_price <= 0:
        raise CliUsageError("max price must be greater than 0")
    if args.min_price > args.max_price:
        raise CliUsageError("min price must not exceed max price")
    if args.min_volume < 0:
        raise CliUsageError("min volume must be at least 0")
    if args.max_pe <= 0:
        raise CliUsageError("max P/E must be greater than 0")
    if args.min_market_cap < 0:
        raise CliUsageError("min market cap must be at least 0")
    if args.min_current_ratio < 0:
        raise CliUsageError("min current ratio must be at least 0")
    if args.max_debt_equity < 0:
        raise CliUsageError("max debt/equity must be at least 0")
    if not 0 <= args.max_decline_from_high <= 1:
        raise CliUsageError("max decline from high must be between 0 and 1")


def _missing_configuration(environ: Mapping[str, str]) -> list[str]:
    return [key for key in REQUIRED_ENVIRONMENT if not environ.get(key, "").strip()]


@contextmanager
def _daily_lock(state_dir: Path) -> Iterator[None]:
    state_dir = state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = (state_dir / LOCK_FILENAME).resolve()
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockBusyError("daily analysis is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _default_flow_factory(preferences: object) -> AnalysisFlow:
    from stockbot.flows.undervalued import UndervaluedAnalysisFlow

    return UndervaluedAnalysisFlow(preferences)  # type: ignore[arg-type]


def _preferences_from_args(args: argparse.Namespace) -> object:
    from stockbot.flows.undervalued import ValueScreeningPreferences

    return ValueScreeningPreferences(
        max_price=args.max_price,
        min_price=args.min_price,
        min_volume=args.min_volume,
        max_pe=args.max_pe,
        min_market_cap=args.min_market_cap,
        min_current_ratio=args.min_current_ratio,
        max_debt_equity=args.max_debt_equity,
        price_vs_high=args.max_decline_from_high,
    )


def _flow_worker(
    connection: object,
    flow_factory: Callable[[object], AnalysisFlow],
    preferences: object,
    universe: list[Ticker],
) -> None:
    """Run the expensive flow in an isolated child and return one result."""
    pipe = connection

    def cancel_on_sigterm(_signum: int, _frame: object) -> None:
        raise asyncio.CancelledError("analysis worker received SIGTERM")

    signal.signal(signal.SIGTERM, cancel_on_sigterm)
    try:
        flow = flow_factory(preferences)
        report = asyncio.run(flow.execute_undervalued_analysis(universe=universe))
        pipe.send(("report", report))  # type: ignore[attr-defined]
    except BaseException as exc:
        try:
            pipe.send(("error", f"{type(exc).__name__}: {exc}"))  # type: ignore[attr-defined]
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        pipe.close()  # type: ignore[attr-defined]


def _run_flow_supervised(
    flow_factory: Callable[[object], AnalysisFlow],
    preferences: object,
    universe: list[Ticker],
    timeout: float,
) -> str:
    """Supervise the production flow with a hard Linux process deadline."""
    context = multiprocessing.get_context("fork")
    parent_pipe, child_pipe = context.Pipe(duplex=False)
    worker = context.Process(
        target=_flow_worker,
        args=(child_pipe, flow_factory, preferences, universe),
        name="stockbot-daily-analysis",
    )
    worker.start()
    child_pipe.close()
    try:
        if parent_pipe.poll(timeout):
            kind, value = parent_pipe.recv()
            worker.join(timeout=0.5)
            if kind == "report":
                return value
            raise RuntimeError(f"analysis worker failed: {value}")

        worker.terminate()
        worker.join(timeout=0.5)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=0.5)
        raise RuntimeError(f"analysis timed out after {timeout:g} seconds")
    finally:
        parent_pipe.close()
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=0.5)


def _utc_timestamp(value: datetime) -> tuple[datetime, str, str]:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    generated_at = value.isoformat(timespec="seconds").replace("+00:00", "Z")
    filename_stamp = value.strftime("%Y%m%dT%H%M%SZ")
    return value, generated_at, filename_stamp


def _rotating_symbol_window(
    universe: Sequence[Ticker], max_symbols: int, selection_date: date
) -> list[Ticker]:
    """Return a date-rotated window from a stable, de-duplicated symbol list."""
    by_symbol: dict[str, Ticker] = {}
    for ticker in sorted(
        universe, key=lambda item: (item.symbol, item.source, item.exchange)
    ):
        by_symbol.setdefault(ticker.symbol, ticker)
    stable_universe = list(by_symbol.values())
    total = len(stable_universe)
    if max_symbols == 0 or max_symbols >= total:
        return stable_universe

    start = (selection_date.toordinal() * max_symbols) % total
    return [stable_universe[(start + offset) % total] for offset in range(max_symbols)]


def _reserve_collision_safe_path(output_dir: Path, filename_stamp: str) -> Path:
    """Atomically claim a generated filename and return its placeholder path."""
    base = f"daily_undervalued_{filename_stamp}"
    suffix = 0
    while True:
        suffix_text = "" if suffix == 0 else f"_{suffix}"
        candidate = output_dir / f"{base}{suffix_text}.md"
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            suffix += 1
            continue
        os.close(descriptor)
        return candidate


def _validate_output_path(report_path: Path, state_dir: Path) -> None:
    canonical_report = report_path.expanduser().resolve()
    canonical_state = state_dir.expanduser().resolve()
    if canonical_report == canonical_state or canonical_report.is_relative_to(
        canonical_state
    ):
        raise RuntimeError("output path is inside the protected state directory")


def _write_temporary_report(report_path: Path, encoded: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=report_path.parent,
        prefix=f".{report_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(encoded)
        temporary.flush()
        os.fsync(temporary.fileno())
    return temporary_path


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(report_path: Path, report: str) -> int:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = report.encode("utf-8")
    temporary_path: Path | None = None
    try:
        temporary_path = _write_temporary_report(report_path, encoded)
        os.replace(temporary_path, report_path)
        temporary_path = None
        _fsync_directory(report_path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(encoded)


def _run_daily(
    args: argparse.Namespace,
    *,
    flow_factory: Callable[[object], AnalysisFlow],
    state_dir: Path,
    now: Callable[[], datetime],
    universe_loader: Callable[[], list[Ticker]],
    supervise_flow: bool = False,
) -> dict[str, object]:
    canonical_state_dir = state_dir.expanduser().resolve()
    with _daily_lock(canonical_state_dir):
        run_datetime, generated_at, filename_stamp = _utc_timestamp(now())
        selection_date = run_datetime.date()
        if args.output_file is not None:
            requested_report_path = args.output_file.expanduser().resolve()
            _validate_output_path(requested_report_path, canonical_state_dir)
        else:
            requested_output_dir = (
                args.output_dir or DEFAULT_OUTPUT_DIR
            ).expanduser().resolve()
            _validate_output_path(requested_output_dir, canonical_state_dir)
        loaded_universe = universe_loader()
        if args.universe == "full":
            selected_universe = loaded_universe
        else:
            source = UNIVERSE_SOURCES[args.universe]
            selected_universe = [
                ticker for ticker in loaded_universe if ticker.source == source
            ]
        if not selected_universe:
            raise RuntimeError(f"selected universe '{args.universe}' is empty")
        source_universe = _rotating_symbol_window(selected_universe, 0, selection_date)
        selected_universe = _rotating_symbol_window(
            source_universe, args.max_symbols, selection_date
        )

        preferences = _preferences_from_args(args)
        if supervise_flow:
            report = _run_flow_supervised(
                flow_factory, preferences, selected_universe, args.timeout
            )
        else:
            flow = flow_factory(preferences)
            async def execute_injected_flow() -> str:
                try:
                    return await asyncio.wait_for(
                        flow.execute_undervalued_analysis(universe=selected_universe),
                        timeout=args.timeout,
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"analysis timed out after {args.timeout:g} seconds"
                    ) from exc

            report = asyncio.run(execute_injected_flow())
        if not isinstance(report, str) or not report.strip():
            raise RuntimeError("analysis returned an empty report")

        reserved_generated_path = False
        if args.output_file is not None:
            report_path = requested_report_path
        else:
            requested_output_dir.mkdir(parents=True, exist_ok=True)
            report_path = _reserve_collision_safe_path(
                requested_output_dir, filename_stamp
            )
            reserved_generated_path = True
        try:
            report_bytes = _atomic_write(report_path, report)
        except Exception:
            if reserved_generated_path:
                report_path.unlink(missing_ok=True)
            raise

    return {
        "status": "ok",
        "report_path": str(report_path),
        "run_type": RUN_TYPE,
        "generated_at": generated_at,
        "report_bytes": report_bytes,
        "universe": args.universe,
        "universe_size": len(selected_universe),
        "max_symbols": args.max_symbols,
        "source_universe_size": len(source_universe),
        "selected_universe_size": len(selected_universe),
        "selection_date": selection_date.isoformat(),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    flow_factory: Callable[[object], AnalysisFlow] | None = None,
    state_dir: Path = DEFAULT_STATE_DIR,
    now: Callable[[], datetime] | None = None,
    load_environment: Callable[[], object] = load_dotenv,
    universe_loader: Callable[[], list[Ticker]] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    supervise_flow: bool | None = None,
) -> int:
    """Run the CLI and return a process exit code."""
    standard_out = stdout or sys.stdout
    standard_error = stderr or sys.stderr
    try:
        args = build_parser().parse_args(argv)
        _validate_daily_arguments(args)
    except CliUsageError as exc:
        print(f"error: {exc}", file=standard_error)
        return EXIT_USAGE

    try:
        load_environment()
    except Exception as exc:
        print(f"error: environment loading failed: {exc}", file=standard_error)
        return EXIT_RUNTIME
    active_environment = os.environ if environ is None else environ
    missing = _missing_configuration(active_environment)
    if missing:
        print(
            f"error: missing required configuration: {', '.join(missing)}",
            file=standard_error,
        )
        return EXIT_CONFIG

    try:
        if universe_loader is None:
            from stockbot.screening.universe import load_universe

            universe_loader = load_universe
        payload = _run_daily(
            args,
            flow_factory=flow_factory or _default_flow_factory,
            state_dir=state_dir,
            now=now or (lambda: datetime.now(timezone.utc)),
            universe_loader=universe_loader,
            supervise_flow=(flow_factory is None) if supervise_flow is None else supervise_flow,
        )
    except LockBusyError as exc:
        print(f"error: {exc}", file=standard_error)
        return EXIT_LOCKED
    except Exception as exc:
        print(f"error: daily analysis failed: {exc}", file=standard_error)
        return EXIT_RUNTIME

    marker = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    print(f"STOCKBOT_RESULT_JSON={marker}", file=standard_out)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
