import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import psutil
import torch

from scripts.train_queue import (
    Job, Plan, launch_args, load_commands, mini_complete, parse_command,
    plan_job, queue_lock, run_queue, stop_child,
)
from src.evaluation.final_mini import evaluate_saved_final_mini, save_final_mini_scores
from src.misc.resume_ckpt import checkpoint_info, find_latest_ckpt


class MiniProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def run_full_test_sets_eval(self, **kwargs):
        self.calls.append((kwargs, self.training, torch.is_grad_enabled(), self.weight.item()))


class TestTrainQueue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        dataset = self.root / "dataset" / "interp_12Hz_trainval"
        dataset.mkdir(parents=True)
        (dataset / "bins_val_3.2m.json").write_text(json.dumps({"bins": [f"scene-{i}" for i in range(28)]}))
        cfg = {
            "dataset": {"name": "omniscene", "roots": [str(dataset.parent)], "test_split": "total"},
            "model": {"encoder": {"num_refine": 1}}, "optimizer": {"lr": 1e-4},
            "loss": {"mse": {}}, "seed": 123,
            "trainer": {"max_steps": 10},
            "train": {"eval_final_mini": True, "final_mini_only": False, "use_dynamic_mask": True},
            "data_loader": {stage: {"batch_size": 1, "seed": 123} for stage in ("train", "val", "test")},
            "checkpointing": {"pretrained_model": "init.ckpt", "pretrained_depth": None, "resume_update_module": None},
            "wandb": {"mode": "offline"},
        }
        self.job = Job("test", ["python", "-m", "src.main"], {}, cfg, self.root / "output", 10)

    def manifest(self):
        self.job.output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(self.job.cfg), self.job.output / "queue_config.yaml")

    def checkpoint(self, step, full=True, final=False, backup=False):
        self.manifest()
        folder = self.job.output / ("checkpoints_backups" if backup else "checkpoints")
        folder.mkdir(exist_ok=True)
        path = folder / (f"final-step_{step}.ckpt" if final else f"epoch_0-step_{step}.ckpt")
        payload = {"state_dict": {"weight": torch.tensor(7.)}, "global_step": step}
        if full:
            payload.update(optimizer_states=[{"state": {0: {}}, "param_groups": [{"params": [0]}]}],
                           lr_schedulers=[{"last_epoch": step}], loops={"fit_loop": {"progress": step}})
        torch.save(payload, path)
        return path

    def scores(self):
        save_final_mini_scores(self.job.metrics,
            {name: [0.5, 0.7] for name in ("psnr", "ssim", "lpips", "pcc")}, 2,
            {"checkpoint": str(self.job.final_checkpoint), "global_step": 10,
             "num_refine": self.job.cfg["model"]["encoder"]["num_refine"],
             "target_views": 18, "scenes": ["scene-0", "scene-14"]})

    def test_commands_and_environment(self):
        path = self.root / "train.sh"
        path.write_text("#!/bin/bash\nset -euo pipefail\n# ignored\nCUDA_VISIBLE_DEVICES=0 bash scripts/omniscene_view6_112x200_base_init.sh \\\n_1 # trailing comment\nbash scripts/omniscene_view6_112x200_base_refine.sh _1\n")
        commands = load_commands(path)
        self.assertEqual(len(commands), 2)
        argv, env = parse_command(commands[0])
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(argv[-1], "_1")
        self.assertTrue(Path(argv[1]).is_absolute())
        argv, _ = parse_command('python -m src.main +experiment=omniscene_112x200 "output_dir=with space"')
        self.assertEqual(argv[0], sys.executable)
        for bad in (commands[1] + " && echo done", "echo hello", commands[1] + " --multirun"):
            with self.assertRaises(ValueError):
                parse_command(bad)

    def test_fresh_including_interruption_before_first_checkpoint(self):
        self.assertEqual(plan_job(self.job).action, "fresh")
        self.manifest()
        (self.job.output / "main.log").touch()
        self.assertEqual(plan_job(self.job).action, "fresh")

    def test_resume_skips_weights_only_and_corrupt_files_and_uses_backups(self):
        complete = self.checkpoint(3)
        self.checkpoint(4, full=False, final=True)
        broken = complete.with_name("epoch_0-step_5.ckpt")
        broken.write_bytes(b"incomplete save")
        self.assertEqual(find_latest_ckpt(complete.parent), complete)
        latest = self.checkpoint(6, backup=True)
        plan = plan_job(self.job)
        self.assertEqual((plan.action, plan.checkpoint), ("resume", latest))
        self.assertTrue(checkpoint_info(latest)["resumable"])

    def test_finished_training_requires_complete_mini(self):
        self.checkpoint(10, full=False, final=True)
        self.assertEqual(plan_job(self.job).action, "evaluate")
        self.scores()
        self.assertEqual(plan_job(self.job).action, "skip")
        (self.job.metrics / "scores_pcc_all.json").write_text('[0.5]')
        self.assertEqual(plan_job(self.job).action, "evaluate")

    def test_mini_rejects_wrong_scene_checkpoint_nonfinite_and_stale_scores(self):
        self.checkpoint(10, full=False, final=True)
        self.scores()
        metadata = json.loads((self.job.metrics / "evaluation.json").read_text())
        for key, value in (("scenes", ["wrong", "scene-14"]), ("checkpoint", "/wrong/final-step_10.ckpt"),
                           ("global_step", 9), ("num_refine", 0), ("target_views", 6)):
            (self.job.metrics / "evaluation.json").write_text(json.dumps({**metadata, key: value}))
            self.assertFalse(mini_complete(self.job))
        (self.job.metrics / "evaluation.json").write_text(json.dumps(metadata))
        (self.job.metrics / "scores_pcc_all.json").write_text('[NaN, 0.7]')
        self.assertFalse(mini_complete(self.job))
        self.scores()
        os.utime(self.job.metrics / "scores_all_avg.json", (1, 1))
        self.assertFalse(mini_complete(self.job))

    def test_full_last_step_can_recover_missing_final_callback(self):
        path = self.checkpoint(10)
        self.assertEqual(plan_job(self.job), Plan("evaluate", path))
        model = MiniProbe()
        evaluate_saved_final_mini(model, object(), path, 10, self.job.output, device="cpu")
        self.assertTrue(self.job.final_checkpoint.is_file())
        self.assertEqual(model.calls[0][1:], (False, False, 7.0))

    def test_weights_only_partial_does_not_silently_restart(self):
        self.checkpoint(4, full=False, final=True)
        with self.assertRaisesRegex(ValueError, "不能完整续训"):
            plan_job(self.job)

    def test_mismatched_step_is_invalid(self):
        path = self.checkpoint(3)
        path.rename(path.with_name("epoch_0-step_4.ckpt"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            checkpoint_info(path.with_name("epoch_0-step_4.ckpt"))

    def test_identity_protects_existing_experiments(self):
        self.checkpoint(3)
        self.job.cfg["wandb"]["mode"] = "online"
        self.assertEqual(plan_job(self.job).action, "resume")
        self.job.cfg["optimizer"]["lr"] = 2e-4
        with self.assertRaisesRegex(ValueError, "训练配置"):
            plan_job(self.job)

    def test_init_source_change_is_not_a_resume(self):
        self.checkpoint(3)
        self.job.cfg["checkpointing"]["pretrained_model"] = "another-init.ckpt"
        with self.assertRaisesRegex(ValueError, "初始权重来源"):
            plan_job(self.job)

    def test_single_update_does_not_resume_a_two_update_experiment(self):
        self.job.cfg["model"]["encoder"]["num_refine"] = 2
        self.checkpoint(3)
        self.job.cfg["model"]["encoder"]["num_refine"] = 1
        with self.assertRaisesRegex(ValueError, "训练配置"):
            plan_job(self.job)

    def test_missing_config_is_not_silently_adopted(self):
        self.checkpoint(3)
        (self.job.output / "queue_config.yaml").unlink()
        with self.assertRaisesRegex(ValueError, "缺少可核验配置"):
            plan_job(self.job)

    def test_legacy_wandb_config_can_be_adopted(self):
        self.checkpoint(3)
        (self.job.output / "queue_config.yaml").unlink()
        config = self.job.output / "wandb" / "run-legacy" / "files" / "config.yaml"
        config.parent.mkdir(parents=True)
        OmegaConf.save(OmegaConf.create({k: {"value": v} for k, v in self.job.cfg.items()}), config)
        self.assertEqual(plan_job(self.job).action, "resume")

    def test_launch_resumes_optimizer_instead_of_reinitializing_base(self):
        path = self.checkpoint(3)
        args = launch_args(self.job, Plan("resume", path))
        self.assertIn(f"checkpointing.load={path}", args)
        self.assertIn("checkpointing.pretrained_model=null", args)
        self.assertIn("checkpointing.pretrained_depth=null", args)
        self.assertNotIn("train.final_mini_only=true", args)
        self.assertIn("train.final_mini_only=true", launch_args(self.job, Plan("evaluate", path)))

    def test_final_mini_recovery_loads_exact_weights_without_training(self):
        path = self.checkpoint(10, full=False, final=True)
        model = MiniProbe()
        evaluate_saved_final_mini(model, object(), path, 10, self.job.output, device="cpu")
        kwargs, training, grad, value = model.calls[0]
        self.assertEqual((kwargs["eval_step"], training, grad, value), (10, False, False, 7.0))
        with self.assertRaisesRegex(ValueError, "exact final-step"):
            evaluate_saved_final_mini(model, object(), path, 11, self.job.output, device="cpu")

    def test_dry_run_does_not_create_outputs_or_run_training(self):
        with patch("scripts.train_queue.resolve_job", return_value=self.job), patch("scripts.train_queue.run_training") as run:
            run_queue(["test"], dry_run=True)
            run.assert_not_called()
        self.assertFalse(self.job.output.exists())

    def test_queue_retries_using_new_checkpoint_and_stops_on_interrupt(self):
        plans = []
        def first_run(job, plan, *args):
            plans.append(plan)
            if len(plans) == 1:
                self.checkpoint(3)
                return 1
            raise KeyboardInterrupt
        with patch("scripts.train_queue.resolve_job", return_value=self.job), patch(
            "scripts.train_queue.assert_not_running"
        ), patch("scripts.train_queue.run_training", side_effect=first_run) as run:
            with self.assertRaises(KeyboardInterrupt):
                run_queue(["test", "never-start"], sleep_sec=0)
            self.assertEqual(run.call_count, 2)
        self.assertEqual([p.action for p in plans], ["fresh", "resume"])
        self.assertEqual(plans[1].checkpoint.name, "epoch_0-step_3.ckpt")
        with patch("scripts.train_queue.resolve_job", return_value=self.job), patch(
            "scripts.train_queue.assert_not_running"
        ), patch("scripts.train_queue.run_training", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "重试次数已用尽"):
                run_queue(["test"], max_retries=0)
        self.assertEqual(plan_job(self.job).action, "resume")

    def test_incomplete_optimizer_state_is_not_resumable(self):
        path = self.checkpoint(3)
        state = torch.load(path, weights_only=False)
        state["optimizer_states"] = [{}]
        torch.save(state, path)
        self.assertFalse(checkpoint_info(path)["resumable"])
        with self.assertRaisesRegex(ValueError, "不能完整续训"):
            plan_job(self.job)

    def test_queue_skips_completed_job_and_rejects_zero_exit_partial(self):
        self.checkpoint(10, full=False, final=True)
        self.scores()
        with patch("scripts.train_queue.resolve_job", return_value=self.job), patch("scripts.train_queue.run_training") as run:
            run_queue(["test"])
            run.assert_not_called()
        self.job.final_checkpoint.unlink()
        with patch("scripts.train_queue.resolve_job", return_value=self.job), patch(
            "scripts.train_queue.assert_not_running"
        ), patch("scripts.train_queue.run_training", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "提前结束"):
                run_queue(["test"])

    def test_lock_rejects_duplicate_queue(self):
        with patch("scripts.train_queue.ROOT", self.root), queue_lock():
            with self.assertRaisesRegex(RuntimeError, "已有训练队列"):
                with queue_lock():
                    pass

    def test_cleanup_only_owns_its_process_group(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); print(p.pid, flush=True); time.sleep(60)"],
            start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        try:
            worker = psutil.Process(int(child.stdout.readline()))
            stop_child(child, grace=0.2)
            self.assertIsNotNone(child.poll())
            self.assertTrue(not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE)
        finally:
            if child.poll() is None:
                stop_child(child, grace=0.2)
            child.stdout.close()


if __name__ == "__main__":
    unittest.main()
