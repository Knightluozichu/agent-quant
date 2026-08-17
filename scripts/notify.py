"""Bark 推送通知模块 (iPhone).

Bark 是 iOS 上的开源推送 App:
  1. App Store 搜索 "Bark" 安装
  2. 打开 App, 首页显示你的设备 Key (一串字符)
  3. 用 `uv run python scripts/live_signal.py --set-bark` 配置 (交互式输入, 不进 shell history)
     或: `BARK_KEY=xxx uv run python scripts/live_signal.py --set-bark`

支持环境变量 BARK_KEY 覆盖配置文件.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, cast

LIVE_DIR = Path(__file__).parent.parent / "data" / "live"
CONFIG_FILE = LIVE_DIR / "config.json"
FAILURE_LOG = LIVE_DIR / "notify_failures.log"

BARK_API = "https://api.day.app"
DEFAULT_SOUND = "minuet"
DEFAULT_GROUP = "七星V3"


def load_config() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                payload = json.load(f)
            return cast("dict[str, Any]", payload) if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(cfg: dict[str, Any]) -> None:
    """Persist config atomically so concurrent requests never read a partial JSON file."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    metadata: tuple[int, int, int] | None = None
    try:
        info = CONFIG_FILE.stat()
        metadata = info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)
    except OSError:
        pass

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=LIVE_DIR,
            prefix=f".{CONFIG_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        if metadata is not None and os.geteuid() == 0:
            os.chown(temp_path, metadata[0], metadata[1])
        temp_path.chmod(metadata[2] if metadata is not None else 0o600)
        temp_path.replace(CONFIG_FILE)

        try:
            dir_fd = os.open(LIVE_DIR, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def get_bark_key() -> str | None:
    """优先环境变量, 其次配置文件."""
    key = os.environ.get("BARK_KEY", "").strip()
    if key:
        return key
    value = load_config().get("bark_key")
    return value if isinstance(value, str) else None


def set_bark_key(key: str) -> None:
    cfg = load_config()
    cfg["bark_key"] = key.strip()
    save_config(cfg)


def _record_failure(title: str, error: Exception) -> None:
    """推送最终失败后落盘一行 (时间戳+标题+错误摘要), 供 health_check 巡检.

    不含 Bark key / 完整 URL; 任何写失败都静默, 不影响调用方.
    """
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = f"{type(error).__name__}: {error}"[:200]
        with open(FAILURE_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{title}\t{summary}\n")
    except OSError:
        pass


def push_bark(
    title: str,
    body: str,
    level: str = "active",
    sound: str = DEFAULT_SOUND,
    group: str = DEFAULT_GROUP,
    url: str | None = None,
) -> bool:
    """推送一条 Bark 通知.

    Args:
        title: 标题
        body: 正文 (支持多行)
        level: active(默认) / timeSensitive(突破勿扰) / passive(静默)
        sound: 提示音
        group: 分组 (Bark 里按组折叠)
        url: 点击通知跳转的链接

    Returns:
        是否推送成功
    """
    key = get_bark_key()
    if not key:
        print("  ⚠️  未配置 Bark Key, 跳过推送")
        print("     配置方法: uv run python scripts/live_signal.py --set-bark")
        return False

    payload: dict[str, Any] = {
        "title": title,
        "body": body,
        "level": level,
        "sound": sound,
        "group": group,
        "isArchive": 1,
    }
    if url:
        payload["url"] = url

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BARK_API}/{key}",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    # 重试策略: 最多 max_retries 次, 每次固定超时 10s; 网络类错误重试间隔指数退避 2s → 4s
    # (最坏阻塞 10+2+10+4+10 = 36s); 4xx / Bark 业务失败 (key/参数错误) 不重试
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        last_error: Exception
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("code") == 200:
                    return True
                # Bark 业务失败 (多为 key/参数错误), 重试无意义
                print(f"  ⚠️  Bark 返回异常: {result}")
                return False
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                # 配置错误, 重试无意义
                print(f"  ⚠️  Bark 推送被拒 (HTTP {e.code}, 请检查 Bark Key 配置)")
                return False
            last_error = e  # 5xx 等服务端错误, 退避重试
        except Exception as e:
            last_error = e  # 网络错误/超时/响应非 JSON, 退避重试
        if attempt < max_retries:
            wait = 2 ** attempt
            print(
                f"  ⚠️  Bark 推送失败 (尝试 {attempt}/{max_retries}): "
                f"{last_error}, {wait}s 后重试"
            )
            time.sleep(wait)
        else:
            print(f"  ⚠️  Bark 推送最终失败 (已重试 {max_retries} 次): {last_error}")
            _record_failure(title, last_error)
            return False
    return False  # 不可达 (循环内必然 return), 仅为满足 mypy return 检查


def test_push() -> bool:
    """发送一条测试通知, 验证配置是否正确."""
    key = get_bark_key()
    if not key:
        print("  ❌ 尚未配置 Bark Key")
        print("     配置方法: uv run python scripts/live_signal.py --set-bark")
        return False
    print(f"  正在推送到 Bark (Key: {key[:6]}***)...")
    ok = push_bark(
        title="✅ 七星V3 通知已连通",
        body="恭喜! 实盘信号推送配置成功。\n每个交易日调仓时, 你会在这里收到买卖指令。",
        level="timeSensitive",
        sound="alarm",
    )
    if ok:
        print("  ✓ 推送成功, 请查看 iPhone 通知")
    return ok
