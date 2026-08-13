"""从生产服务器同步实盘状态，并在本地执行只读盘中预演。

默认只从服务器读取三个非秘密文件：
  - data/live/state.json
  - data/live/strategy_mode.json
  - deploy/v4_release_20260811.json

脚本不会读取 config.json/.env，不会写服务器，也不会提交订单。同步前会校验
V4 发布身份与本地生产核心文件哈希；本地旧状态会保存到带时间戳的备份目录。
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import live_signal
import qixing_v4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_LIVE_DIR = PROJECT_ROOT / "data" / "live"
DEFAULT_SERVER = os.environ.get("QIXING_SERVER_SSH", "root@106.52.243.51")
DEFAULT_REMOTE_DIR = os.environ.get("QIXING_REMOTE_DIR", "/opt/quant")
REMOTE_RELEASE_FILE = "deploy/v4_release_20260811.json"
PREVIEW_CORE_FILES = (
    "scripts/qixing_v4.py",
    "scripts/live_signal.py",
    "scripts/run_qixing_v3.py",
    "scripts/risk_overrides.py",
)
VALID_MODES = {"V3-G", "V4_SHADOW", "V4"}
_SERVER_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
_REMOTE_DIR_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")


@dataclass(frozen=True)
class RemoteSnapshot:
    """Validated inputs fetched from one production host."""

    state: dict[str, Any]
    mode: dict[str, Any]
    release: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取有效 JSON: {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} 顶层必须是 JSON object")
    return payload


def _validate_remote(server: str, remote_dir: str) -> None:
    if not _SERVER_RE.fullmatch(server):
        raise ValueError("服务器地址只允许 user@host、主机名或 IPv4")
    if not _REMOTE_DIR_RE.fullmatch(remote_dir) or ".." in Path(remote_dir).parts:
        raise ValueError("远端目录必须是无空格、无 .. 的绝对路径")


def fetch_remote_snapshot(
    server: str,
    remote_dir: str,
    *,
    timeout_seconds: int = 10,
) -> RemoteSnapshot:
    """Use OpenSSH scp in batch mode; the remote side is strictly read-only."""
    _validate_remote(server, remote_dir)
    scp = shutil.which("scp")
    if scp is None:
        raise RuntimeError("本机未找到 scp，无法读取服务器状态")

    remote_root = remote_dir.rstrip("/")
    sources = (
        f"{server}:{remote_root}/data/live/state.json",
        f"{server}:{remote_root}/data/live/strategy_mode.json",
        f"{server}:{remote_root}/{REMOTE_RELEASE_FILE}",
    )
    with tempfile.TemporaryDirectory(prefix="qixing-server-state-") as tmp:
        temp_dir = Path(tmp)
        command = [
            scp,
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout_seconds}",
            *sources,
            str(temp_dir),
        ]
        try:
            result = subprocess.run(  # noqa: S603 - argv only; target/path validated above
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 20,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"连接服务器超时（{timeout_seconds} 秒）") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or "scp 返回非零状态"
            raise RuntimeError(f"读取服务器状态失败: {detail}")

        return RemoteSnapshot(
            state=_load_json(temp_dir / "state.json"),
            mode=_load_json(temp_dir / "strategy_mode.json"),
            release=_load_json(temp_dir / Path(REMOTE_RELEASE_FILE).name),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"服务器状态字段 {field} 必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"服务器状态字段 {field} 必须是有限数")
    return number


def validate_snapshot(
    snapshot: RemoteSnapshot,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate account schema, V4 identity, and local/remote production hashes."""
    mode = str(snapshot.mode.get("mode", "")).strip().upper()
    if mode not in VALID_MODES:
        raise ValueError(f"服务器策略模式无效: {mode or '(empty)'}")

    release = snapshot.release
    if release.get("strategy_mode") != mode:
        raise ValueError(
            f"服务器模式与发布清单不一致: mode={mode}, release={release.get('strategy_mode')}"
        )
    if release.get("strategy_id") != qixing_v4.STRATEGY_ID:
        raise ValueError("服务器发布清单的 V4 策略 ID 与本地不一致")
    if release.get("config_hash") != qixing_v4.CONFIG_HASH:
        raise ValueError("服务器发布清单的 V4 配置哈希与本地不一致")

    release_files = release.get("files")
    if not isinstance(release_files, dict):
        raise ValueError("服务器发布清单缺少文件哈希")
    for relative in PREVIEW_CORE_FILES:
        local_path = project_root / relative
        expected = release_files.get(relative)
        if not expected:
            raise ValueError(f"服务器发布清单缺少核心文件: {relative}")
        if not local_path.is_file():
            raise ValueError(f"本地缺少生产核心文件: {relative}")
        actual = _sha256(local_path)
        if actual != expected:
            raise ValueError(
                f"本地与服务器核心文件不一致: {relative} "
                f"(local={actual[:12]}, server={str(expected)[:12]})"
            )

    state = copy.deepcopy(snapshot.state)
    initial_capital = _finite_number(state.get("initial_capital"), "initial_capital")
    cash = _finite_number(state.get("cash"), "cash")
    if initial_capital <= 0:
        raise ValueError("服务器 initial_capital 必须大于 0")
    if cash < 0:
        raise ValueError("服务器 cash 不能为负数")

    holding = state.get("holding")
    if holding is not None and holding not in live_signal.ALL_CODES:
        raise ValueError(f"服务器存在未知持仓: {holding}")
    shares = state.get("shares")
    if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0:
        raise ValueError("服务器 shares 必须是非负整数")
    if holding is None and shares != 0:
        raise ValueError("服务器空仓但 shares 非零")
    if holding is not None and shares <= 0:
        raise ValueError("服务器有持仓但 shares 不为正数")

    entry_price = _finite_number(state.get("entry_price", 0.0), "entry_price")
    if holding is not None and entry_price <= 0:
        raise ValueError("服务器有持仓但 entry_price 无效")
    if not isinstance(state.get("trade_log", []), list):
        raise ValueError("服务器 trade_log 必须是列表")
    pending = state.get("pending_order")
    if pending is not None and not isinstance(pending, dict):
        raise ValueError("服务器 pending_order 必须是 object 或 null")

    # V4 上线前的 state 没有 v4_state；本地预演按生产迁移规则冷启动。
    if state.get("v4_state") is None:
        state.pop("v4_state", None)
    normalized = live_signal._normalize_state(state)
    runtime = normalized["v4_state"]
    if runtime.get("strategy_id") != qixing_v4.STRATEGY_ID:
        raise ValueError("服务器 v4_state 策略 ID 与本地不一致")
    if runtime.get("config_hash") != qixing_v4.CONFIG_HASH:
        raise ValueError("服务器 v4_state 配置哈希与本地不一致")
    if not isinstance(runtime.get("candidate_history"), list):
        raise ValueError("服务器 V4 确认历史必须是列表")

    risk_exposure = _finite_number(normalized.get("risk_exposure", 1.0), "risk_exposure")
    if not 0.0 < risk_exposure <= 1.0:
        raise ValueError("服务器 risk_exposure 必须在 (0, 1] 区间")
    return cast("dict[str, Any]", normalized)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f"{path.name}.server-sync.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.chmod(0o600)
    temp.replace(path)


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_snapshot(
    snapshot: RemoteSnapshot,
    *,
    live_dir: Path = LOCAL_LIVE_DIR,
) -> Path:
    """Back up local runtime files and atomically install the remote snapshot."""
    live_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = live_dir / "quant_state.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        backup = live_dir / "server_sync_backups" / stamp
        suffix = 1
        while backup.exists():
            backup = live_dir / "server_sync_backups" / f"{stamp}-{suffix}"
            suffix += 1
        backup.mkdir(parents=True, mode=0o700)
        backup.chmod(0o700)

        for name in ("state.json", "strategy_mode.json"):
            current = live_dir / name
            if current.exists():
                shutil.copy2(current, backup / name)
                (backup / name).chmod(0o600)

        _atomic_write_json(live_dir / "state.json", snapshot.state)
        _atomic_write_json(live_dir / "strategy_mode.json", snapshot.mode)
        _fsync_dir(live_dir)
        return backup
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _run_live_signal(*arguments: str) -> int:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "live_signal.py"), *arguments]
    sys.stdout.flush()
    return subprocess.run(  # noqa: S603 - fixed local interpreter and script path
        command, cwd=PROJECT_ROOT, check=False
    ).returncode


def run_preview(*, sync_market_data: bool = True) -> int:
    """Refresh completed bars, then ensure dry-run leaves account state untouched."""
    if sync_market_data:
        print("\n[2/3] 补齐本地已完成交易日日线")
        if _run_live_signal("--sync-only") != 0:
            raise RuntimeError("本地行情同步失败，未执行盘中预演")

    state_path = LOCAL_LIVE_DIR / "state.json"
    mode_path = LOCAL_LIVE_DIR / "strategy_mode.json"
    before = (_sha256(state_path), _sha256(mode_path))
    print("\n[3/3] 使用服务器真实仓位执行本地盘中预演")
    result = _run_live_signal("--dry-run")
    after = (_sha256(state_path), _sha256(mode_path))
    if before != after:
        raise RuntimeError("安全校验失败：dry-run 改动了本地账户或策略模式")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读同步服务器实盘状态，并在本地执行 V4 盘中预演")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="SSH 目标，默认生产服务器")
    parser.add_argument(
        "--remote-dir", default=DEFAULT_REMOTE_DIR, help="服务器项目目录，默认 /opt/quant"
    )
    parser.add_argument("--timeout", type=int, default=10, help="SSH 连接超时秒数")
    parser.add_argument(
        "--state-only", action="store_true", help="只同步账户状态，不更新行情、不做预演"
    )
    parser.add_argument(
        "--skip-data-sync", action="store_true", help="跳过已完成交易日日线同步，直接盘中预演"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0:
        print("❌ --timeout 必须大于 0")
        return 2

    print("[1/3] 只读拉取服务器账户状态、策略模式和 V4 发布清单")
    print(f"      来源: {args.server}:{args.remote_dir}")
    try:
        snapshot = fetch_remote_snapshot(args.server, args.remote_dir, timeout_seconds=args.timeout)
        normalized = validate_snapshot(snapshot)
        backup = install_snapshot(snapshot)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"❌ 同步中止: {exc}")
        return 1

    holding = normalized.get("holding")
    holding_text = (
        f"{holding} {live_signal.name_of(holding)} {normalized['shares']}份" if holding else "空仓"
    )
    print(f"      模式: {snapshot.mode['mode']} ({qixing_v4.STRATEGY_ID})")
    print(f"      持仓: {holding_text}")
    print(f"      现金: {normalized['cash']:,.2f} 元")
    print(f"      状态版本: {normalized.get('_version', 0)}")
    print(f"      本地旧状态备份: {backup}")
    print("      安全边界: 未读取服务器 config/.env，未写服务器，未连接券商")

    if args.state_only:
        print("\n✓ 服务器实盘状态已同步到本地")
        return 0

    try:
        result = run_preview(sync_market_data=not args.skip_data_sync)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1
    if result != 0:
        print(f"❌ 盘中预演失败，live_signal 返回 {result}")
        return result
    print("\n✓ 同步与盘中预演完成；账户状态哈希未变化")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
