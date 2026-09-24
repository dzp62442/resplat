#!/usr/bin/env python3
"""顺序执行 train.sh：完成即跳过，中断续训，训练完成但 mini 缺失则补评估。"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

import psutil
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.misc.resume_ckpt import checkpoint_info, find_latest_ckpt


@dataclass
class Job:
    command: str
    argv: list[str]
    env: dict[str, str]
    cfg: dict
    output: Path
    max_steps: int

    @property
    def final_checkpoint(self):
        return self.output / "checkpoints" / f"final-step_{self.max_steps}.ckpt"

    @property
    def metrics(self):
        return self.output / f"mini-final-step_{self.max_steps}" / "metrics"


@dataclass
class Plan:
    action: str
    checkpoint: Path | None = None


def load_commands(path: Path) -> list[str]:
    commands, buffer = [], ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Accept the old list header; no arbitrary shell setup is executed.
        if not buffer and line in ("set -euo pipefail", "set -e"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        commands.append(buffer + line)
        buffer = ""
    if buffer:
        raise ValueError("队列末尾存在未结束的反斜杠续行")
    if not commands:
        raise ValueError("训练队列为空")
    return commands


def parse_command(command: str) -> tuple[list[str], dict[str, str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    tokens = list(lexer)
    env = {}
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        key, value = tokens.pop(0).split("=", 1)
        env[key] = value
    if not tokens or any(token in (";", "&", "&&", "|", "||", ">", ">>", "<") for token in tokens):
        raise ValueError("每条队列项必须是一条训练命令，不支持管道、重定向或 &&；请分行书写")
    if any("$" in token or "`" in token for token in tokens + list(env.values())):
        raise ValueError("队列使用直接参数，不支持 shell 变量展开；请填写实际参数值")
    if tokens[0] == "bash" and len(tokens) > 1:
        script = Path(tokens[1])
        script = script if script.is_absolute() else ROOT / script
        if script.resolve().parent != ROOT / "scripts" or not re.fullmatch(
            r"omniscene_view6_(112x200|224x400)_base_(init|refine)\.sh", script.name
        ):
            raise ValueError(f"不支持的阶段脚本：{script}")
        tokens[1] = str(script)
    elif len(tokens) >= 3 and Path(tokens[0]).name in ("python", "python3") and tokens[1:3] == ["-m", "src.main"]:
        tokens[0] = sys.executable
    else:
        raise ValueError("支持 bash scripts/omniscene_*_base_*.sh 或 python -m src.main")
    if any(token in ("--cfg", "--help", "-h", "--multirun", "-m") for token in tokens[2 if tokens[0] == "bash" else 3:]):
        raise ValueError("队列项只接受单次训练，不接受配置查询或 multirun")
    return tokens, env


def job_environment(job_env: dict) -> dict:
    env = {**os.environ, **job_env}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def resolve_job(command: str) -> Job:
    argv, env = parse_command(command)
    # Ask the actual stage script/Hydra for its final config, not main.yaml alone.
    result = subprocess.run(
        [*argv, "--cfg", "job", "--resolve"], cwd=ROOT,
        env={**job_environment(env), "CUDA_VISIBLE_DEVICES": ""},
        text=True, capture_output=True, timeout=90, check=True,
    )
    cfg = OmegaConf.to_container(OmegaConf.create(result.stdout), resolve=True)
    if cfg["mode"] != "train" or cfg["dataset"]["name"] != "omniscene":
        raise ValueError("当前队列仅支持 OmniScene 训练")
    if not cfg["train"]["eval_final_mini"] or cfg["train"]["final_mini_only"]:
        raise ValueError("队列要求 eval_final_mini=true，原始命令不能设置 final_mini_only")
    if cfg["checkpointing"]["resume"] or cfg["checkpointing"]["load"] is not None:
        raise ValueError("checkpointing.resume/load 由队列管理，请从原始命令中移除")
    if cfg["trainer"]["max_steps"] <= 0 or cfg["data_loader"]["test"]["batch_size"] != 1:
        raise ValueError("需要正整数 max_steps 和 test batch_size=1")
    output = Path(cfg["output_dir"])
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    return Job(command, argv, env, cfg, output, cfg["trainer"]["max_steps"])


def identity(cfg: dict) -> dict:
    """Exclude logging/transport knobs; preserve model, loss, seed and training budget."""
    train = {k: v for k, v in cfg["train"].items() if not k.startswith(
        ("eval_", "diagnostics_", "no_log", "print_log")
    ) and k not in ("final_mini_only", "extended_visualization", "no_viz_video")}
    return {
        **{k: cfg[k] for k in ("dataset", "model", "optimizer", "loss", "seed")},
        "max_steps": cfg["trainer"]["max_steps"],
        "gradient_clip_val": cfg["trainer"].get("gradient_clip_val"), "train": train,
        "loaders": {k: {field: v[field] for field in ("batch_size", "seed")}
                    for k, v in cfg["data_loader"].items()},
    }


def verify_identity(job: Job) -> None:
    manifest = job.output / "queue_config.yaml"
    candidates = [manifest] if manifest.is_file() else sorted(
        job.output.glob("wandb/*run-*/files/config.yaml"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        candidates = list(job.output.glob("mini-final-step_*/config.yaml"))
    if candidates:
        saved = OmegaConf.to_container(OmegaConf.load(candidates[0]), resolve=True)
        if "value" in saved.get("model", {}):
            saved = {k: v["value"] for k, v in saved.items() if isinstance(v, dict) and "value" in v}
        if identity(saved) != identity(job.cfg):
            raise ValueError(f"{job.output.name} 的训练配置与已有实验不同；请使用新后缀，不会自动覆盖")
        # A queue manifest also remembers which Init/pretrained source was requested.
        if manifest.is_file():
            for key in ("pretrained_model", "pretrained_depth", "resume_update_module"):
                if saved["checkpointing"].get(key) != job.cfg["checkpointing"].get(key):
                    raise ValueError(f"{job.output.name} 的初始权重来源 {key} 已改变，请使用新后缀")
    elif any(job.output.glob("checkpoints*/*.ckpt")):
        raise ValueError(f"{job.output} 有权重但缺少可核验配置，停止自动恢复")


def mini_complete(job: Job) -> bool:
    try:
        metadata = json.loads((job.metrics / "evaluation.json").read_text())
        summary = json.loads((job.metrics / "scores_all_avg.json").read_text())
        root = Path(job.cfg["dataset"]["roots"][0])
        split = ROOT / root / "interp_12Hz_trainval" / "bins_val_3.2m.json"
        scenes = json.loads(split.read_text())["bins"][0::14][:2048]
        if not scenes or metadata.get("scenes") != scenes:
            return False
        if (metadata.get("global_step") != job.max_steps
                or metadata.get("num_refine") != job.cfg["model"]["encoder"]["num_refine"]
                or metadata.get("split") != "mini" or metadata.get("target_views") != 18
                or metadata.get("sample_count") != len(scenes)
                or Path(metadata["checkpoint"]).resolve() != job.final_checkpoint.resolve()):
            return False
        if (job.metrics / "scores_all_avg.json").stat().st_mtime_ns < job.final_checkpoint.stat().st_mtime_ns:
            return False
        for name in ("psnr", "ssim", "lpips", "pcc"):
            values = json.loads((job.metrics / f"scores_{name}_all.json").read_text())
            if len(values) != len(scenes) or not all(math.isfinite(v) for v in values):
                return False
            if not math.isfinite(summary[name]) or not math.isclose(
                summary[name], sum(values) / len(values), rel_tol=1e-9, abs_tol=1e-9
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def plan_job(job: Job) -> Plan:
    verify_identity(job)
    if job.final_checkpoint.is_file():
        try:
            if checkpoint_info(job.final_checkpoint)["step"] == job.max_steps:
                return Plan("skip" if mini_complete(job) else "evaluate", job.final_checkpoint)
        except Exception as error:
            print(f"[检查点不可用] {job.final_checkpoint}: {error}", flush=True)
    candidates = []
    for folder in ("checkpoints", "checkpoints_backups"):
        try:
            checkpoint = find_latest_ckpt(job.output / folder)
            candidates.append((checkpoint_info(checkpoint)["step"], checkpoint))
        except ValueError:
            pass
    if candidates:
        step, checkpoint = max(candidates)
        if step > job.max_steps:
            raise ValueError("已有检查点超过目标步数，请核查训练预算")
        if step == job.max_steps:
            return Plan("evaluate", checkpoint)
        return Plan("resume", checkpoint)
    if any(job.output.glob("checkpoints*/*.ckpt")):
        raise ValueError(f"{job.output.name} 只有损坏或仅权重检查点，不能完整续训，也不会自动从头覆盖")
    return Plan("fresh")


def launch_args(job: Job, plan: Plan) -> list[str]:
    args = [*job.argv, "checkpointing.resume=false"]
    if plan.action in ("resume", "evaluate"):
        args += [f"checkpointing.load={plan.checkpoint}", "checkpointing.pretrained_model=null",
                 "checkpointing.pretrained_depth=null", "checkpointing.resume_update_module=null"]
    if plan.action == "evaluate":
        args += ["train.final_mini_only=true"]
    return args


def assert_not_running(job: Job) -> None:
    for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
        try:
            argv = process.info["cmdline"] or []
            if "src.main" not in argv:
                continue
            for token in reversed(argv):
                if token.startswith("output_dir="):
                    path = Path(token.split("=", 1)[1])
                    path = Path(process.info["cwd"] or ROOT) / path
                    if path.resolve() == job.output:
                        raise RuntimeError(f"{job.output.name} 已有训练进程 PID={process.pid}，停止以防覆盖")
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def stop_child(child: subprocess.Popen, grace: float = 30) -> None:
    """Only signal our own new process group, including loader/W&B children."""
    members = []
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) == child.pid and process.status() != psutil.STATUS_ZOMBIE:
                members.append(process)
        except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        child.poll()
        _, members = psutil.wait_procs(members, timeout=grace if sig == signal.SIGINT else 5)
        members = [p for p in members if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
        if not members:
            break
    child.wait(timeout=5)
    if members:
        raise RuntimeError("本次训练的子进程未退出，停止队列以避免重叠运行")


def run_training(job: Job, plan: Plan, exit_timeout: float = 300, poll_interval: float = 10) -> int:
    child = subprocess.Popen(
        launch_args(job, plan), cwd=ROOT,
        env={**job_environment(job.env), "RESPLAT_QUEUE_MANAGED": "1"},
        start_new_session=True,
    )
    completed_at = None
    try:
        while child.poll() is None:
            if job.final_checkpoint.is_file() and mini_complete(job):
                completed_at = completed_at or time.monotonic()
                if time.monotonic() - completed_at >= exit_timeout:
                    print("[收尾超时] 评估结果已保存，回收本次启动的后台进程。", flush=True)
                    stop_child(child)
                    return child.returncode
            else:
                completed_at = None
            try:
                return child.wait(timeout=poll_interval)
            except subprocess.TimeoutExpired:
                pass
        return child.returncode
    finally:
        # Also runs when the queue receives Ctrl-C/SIGTERM: do not advance/retry.
        stop_child(child)


@contextmanager
def queue_lock():
    name = hashlib.sha256(str(ROOT.resolve()).encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"resplat-train-queue-{os.getuid()}-{name}.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("本项目已有训练队列运行，请勿重复启动") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_queue(commands: list[str], dry_run=False, max_retries=3, sleep_sec=30,
              exit_timeout=300, poll_interval=10) -> None:
    seen = set()
    for index, command in enumerate(commands, 1):
        job = resolve_job(command)
        if job.output in seen:
            raise ValueError(f"队列中重复的 output_dir：{job.output}")
        seen.add(job.output)
        retries = 0
        while True:
            plan = plan_job(job)
            print(f"[{index}/{len(commands)} {plan.action}] {job.output.name}"
                  + (f" <- {plan.checkpoint.name}" if plan.checkpoint else ""), flush=True)
            if not dry_run:
                assert_not_running(job)
            if plan.action == "skip":
                break
            if dry_run:
                print(shlex.join(launch_args(job, plan)), flush=True)
                break
            job.output.mkdir(parents=True, exist_ok=True)
            manifest = job.output / "queue_config.yaml"
            if not manifest.exists():
                temporary = manifest.with_suffix(".yaml.tmp")
                OmegaConf.save(OmegaConf.create(job.cfg), temporary)
                temporary.replace(manifest)
            code = run_training(job, plan, exit_timeout, poll_interval)
            if plan_job(job).action == "skip":
                print(f"[完成] {job.output.name} 训练和最终 mini 均已完成", flush=True)
                break
            if code in (-signal.SIGINT, -signal.SIGTERM, 130, 143):
                raise KeyboardInterrupt
            if code == 0:
                raise RuntimeError(f"{job.output.name} 提前结束但未完成（可能触发诊断保护），停止队列")
            if retries >= max_retries:
                raise RuntimeError(f"{job.output.name} 失败，退出码 {code}，重试次数已用尽")
            retries += 1
            print(f"[重试 {retries}/{max_retries}] {sleep_sec}s 后重新检查检查点并恢复", flush=True)
            time.sleep(sleep_sec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default="train.sh", help="相对项目根目录的命令清单")
    parser.add_argument("--dry-run", action="store_true", help="只读检查并打印计划，不创建实验目录或启动训练")
    parser.add_argument("--max-retries", type=int, default=3, help="每条命令的额外重试次数，0 表示不重试")
    parser.add_argument("--sleep", type=float, default=30)
    parser.add_argument("--exit-timeout", type=float, default=300)
    parser.add_argument("--poll-interval", type=float, default=10)
    args = parser.parse_args()
    if args.max_retries < 0 or args.sleep < 0 or args.exit_timeout <= 0 or args.poll_interval <= 0:
        parser.error("重试次数/间隔不能为负，轮询和收尾超时必须大于 0")
    commands = load_commands(ROOT / args.queue)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.dry_run:
            run_queue(commands, dry_run=True)
        else:
            with queue_lock():
                run_queue(commands, max_retries=args.max_retries, sleep_sec=args.sleep,
                          exit_timeout=args.exit_timeout, poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        print("\n[已暂停] 队列不再推进；重新执行本命令即可按最近完整检查点恢复。", flush=True)
        return 130
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"[队列停止] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
