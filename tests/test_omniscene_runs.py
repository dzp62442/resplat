import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.final_mini import save_final_mini_scores
from src.global_cfg import get_cfg, set_cfg
from src.misc.final_checkpoint import FinalCheckpoint
from src.misc.benchmarker import Benchmarker
from src.model.decoder.decoder import DecoderOutput
from src.model.model_wrapper import ModelWrapper


ROOT = Path(__file__).resolve().parents[1]


class StageProbe(LightningModule):
    """Exercise real Lightning completion/checkpoint ordering without GPU work."""

    on_train_end = ModelWrapper.on_train_end
    on_save_checkpoint = ModelWrapper.on_save_checkpoint
    on_load_checkpoint = ModelWrapper.on_load_checkpoint

    def __init__(self, num_refine=0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.train_cfg = SimpleNamespace(eval_final_mini=True)
        self.eval_data_cfg = SimpleNamespace(name="omniscene", test_split="mini")
        self.num_refine = num_refine
        self.eval_cnt = 7
        self._diagnostic_stop_reason = None
        self.evaluations = []

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def run_full_test_sets_eval(self, final_output_dir):
        checkpoint = Path(get_cfg().output_dir) / "checkpoints" / f"final-step_{self.global_step}.ckpt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert torch.equal(state["state_dict"]["weight"], self.weight.detach().cpu())
        self.evaluations.append((self.global_step, self.training, torch.is_grad_enabled()))
        save_final_mini_scores(
            final_output_dir / "metrics",
            {name: [0.5, 0.7] for name in ("psnr", "ssim", "lpips", "pcc")},
            2,
            {"global_step": self.global_step, "num_refine": self.num_refine},
        )


class StopEarly(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        trainer.should_stop = True


class TestOmniSceneRuns(unittest.TestCase):
    def _trainer(self, directory, steps, callbacks=()):
        return Trainer(
            accelerator="cpu", max_steps=steps, logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0,
            callbacks=[FinalCheckpoint(directory / "checkpoints"), *callbacks],
        )

    def test_both_stages_evaluate_exact_final_state_and_resume_counter(self):
        previous_cfg = get_cfg()
        try:
            for num_refine in (0, 1, 2):
                with self.subTest(num_refine=num_refine), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    set_cfg(OmegaConf.create({"output_dir": tmp}))
                    loader = DataLoader(TensorDataset(torch.ones(4)), batch_size=1)
                    model = StageProbe(num_refine)
                    trainer = self._trainer(directory, 2)
                    trainer.fit(model, train_dataloaders=loader)
                    self.assertEqual(model.evaluations, [(2, False, False)])
                    self.assertTrue(model.training)
                    self.assertTrue((directory / "mini-final-step_2/metrics/scores_all_avg.json").is_file())

                    resume_path = directory / "resume.ckpt"
                    trainer.save_checkpoint(resume_path)
                    resumed = StageProbe(num_refine)
                    resumed.eval_cnt = 0
                    self._trainer(directory, 3).fit(
                        resumed, train_dataloaders=loader, ckpt_path=resume_path
                    )
                    self.assertEqual(resumed.eval_cnt, 7)
                    self.assertEqual(resumed.evaluations, [(3, False, False)])
        finally:
            set_cfg(previous_cfg)

    def test_early_stop_does_not_publish_final_mini(self):
        previous_cfg = get_cfg()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                set_cfg(OmegaConf.create({"output_dir": tmp}))
                model = StageProbe()
                self._trainer(directory, 3, [StopEarly()]).fit(
                    model, train_dataloaders=DataLoader(TensorDataset(torch.ones(4)))
                )
                self.assertEqual(model.global_step, 1)
                self.assertEqual(model.evaluations, [])
                self.assertFalse(list(directory.glob("mini-final-*")))
        finally:
            set_cfg(previous_cfg)

    def test_final_mini_requires_complete_finite_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "metrics"
            scores = {name: [0.5, 0.7] for name in ("psnr", "ssim", "lpips", "pcc")}
            scores["pcc"] = [0.5]
            with self.assertRaises(RuntimeError):
                save_final_mini_scores(directory, scores, 2, {})
            self.assertFalse(directory.exists())
            scores["pcc"] = [float("nan"), 0.7]
            with self.assertRaises(RuntimeError):
                save_final_mini_scores(directory, scores, 2, {})
            self.assertFalse(directory.exists())
            scores["pcc"] = [0.5, 0.7]
            result = save_final_mini_scores(directory, scores, 2, {"global_step": 33334})
            self.assertAlmostEqual(result["pcc"], 0.6)
            self.assertEqual(json.loads((directory / "evaluation.json").read_text())["sample_count"], 2)

    def test_final_evaluation_loop_covers_all_samples_and_uses_last_refinement(self):
        previous_cfg = get_cfg()
        depth = torch.arange(18 * 8 * 8, dtype=torch.float32).reshape(1, 18, 8, 8)
        image = torch.full((1, 18, 3, 8, 8), 0.7)
        init_render = DecoderOutput(color=torch.full_like(image, 0.2), depth=depth)
        refined_render = DecoderOutput(color=torch.full_like(image, 0.6), depth=depth)
        batches = [
            {"context": {}, "target": {"image": image, "rel_depth": depth,
             "extrinsics": None, "intrinsics": None, "near": None, "far": None},
             "scene": [f"scene-{i}"]}
            for i in range(2)
        ]
        try:
            for num_refine in (0, 1, 2):
                with self.subTest(num_refine=num_refine), tempfile.TemporaryDirectory() as tmp:
                    set_cfg(OmegaConf.create({"output_dir": tmp}))
                    wrapper = ModelWrapper.__new__(ModelWrapper)
                    LightningModule.__init__(wrapper)
                    wrapper.encoder = torch.nn.Identity()
                    wrapper.encoder.cfg = SimpleNamespace(num_refine=num_refine)
                    wrapper.encoder.forward = Mock(return_value={
                        "gaussians": None, "depths": None, "condition_features": None,
                    })
                    renders = [init_render] * max(num_refine - 1, 0) + [refined_render] if num_refine else []
                    wrapper.encoder.forward_update = Mock(return_value={"render": renders})
                    wrapper.decoder = SimpleNamespace(forward=Mock(return_value=init_render))
                    wrapper.eval_data_cfg = SimpleNamespace(name="omniscene", test_split="mini")
                    wrapper.train_cfg = SimpleNamespace(
                        eval_data_length=1, eval_time_skip_steps=0,
                        eval_depth=False, eval_deterministic=False,
                    )
                    wrapper.test_cfg = SimpleNamespace(inference_window_size=None)
                    wrapper.benchmarker = Benchmarker()
                    wrapper.data_shim = lambda batch: batch
                    wrapper.transfer_batch_to_device = lambda batch, *args, **kwargs: batch
                    logger = Mock()
                    wrapper._trainer = SimpleNamespace(
                        global_step=33334, logger=logger,
                        datamodule=SimpleNamespace(test_dataloader=lambda **kwargs: DataLoader(batches, batch_size=None)),
                    )
                    out = Path(tmp) / "mini-final-step_33334"
                    with patch("src.model.model_wrapper.compute_lpips", return_value=torch.tensor([0.1])), patch(
                        "src.model.model_wrapper.compute_ssim", return_value=torch.tensor([0.8])
                    ):
                        wrapper.run_full_test_sets_eval(final_output_dir=out)
                    scores = json.loads((out / "metrics/scores_psnr_all.json").read_text())
                    self.assertEqual(len(scores), 2)  # ignores periodic eval_data_length=1
                    self.assertAlmostEqual(scores[0], 20.0 if num_refine else 6.0206, places=3)
                    metadata = json.loads((out / "metrics/evaluation.json").read_text())
                    self.assertEqual(metadata["scenes"], ["scene-0", "scene-1"])
                    self.assertEqual(metadata["target_views"], 18)
                    self.assertEqual(metadata["num_refine"], num_refine)
                    logged = logger.log_metrics.call_args.args[0]
                    self.assertAlmostEqual(logged["final_mini/pcc"], 1.0)
                    self.assertEqual(logged["test/psnr"], logged["final_mini/psnr"])
                    # Queue-only recovery must use the same loop without an
                    # attached Trainer and preserve the saved global step.
                    data_module = wrapper._trainer.datamodule
                    wrapper._trainer = None
                    with patch("src.model.model_wrapper.compute_lpips", return_value=torch.tensor([0.1])), patch(
                        "src.model.model_wrapper.compute_ssim", return_value=torch.tensor([0.8])
                    ):
                        wrapper.run_full_test_sets_eval(
                            final_output_dir=out, eval_data_module=data_module,
                            eval_step=33334, eval_logger=logger,
                        )
                    recovered = json.loads((out / "metrics/scores_psnr_all.json").read_text())
                    self.assertEqual(recovered, scores)
                    self.assertEqual(logger.log_metrics.call_args.kwargs["step"], 33334)
        finally:
            set_cfg(previous_cfg)

    def _script(self, resolution, stage, *args):
        # Stub only python: run the real shell parsing without starting training.
        return subprocess.run(
            ["bash", "-c", 'python() { printf "%s\\n" "$@"; }; export -f python; bash "$@"',
             "test", str(ROOT / "scripts" / f"omniscene_view6_{resolution}_base_{stage}.sh"), *args],
            cwd="/tmp", text=True, capture_output=True,
        )

    def test_suffixes_select_matching_stage_paths_and_allow_overrides(self):
        for resolution in ("112x200", "224x400"):
            root = f"checkpoints/resplat/omniscene-view6-{resolution}"
            for stage in ("init", "refine"):
                result = self._script(resolution, stage, "_1", "seed=123", "--cfg", "job")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"output_dir={root}/base-{stage}_1\n", result.stdout)
                self.assertIn("seed=123\n", result.stdout)
                self.assertIn(f"model.encoder.num_refine={int(stage == 'refine')}\n", result.stdout)
                if stage == "refine":
                    self.assertIn(f"checkpointing.pretrained_model={root}/base-init_1/checkpoints/final-step_66667.ckpt\n", result.stdout)
                    self.assertIn("train.refine_raw_scale_regularization_weight=0.01\n", result.stdout)
            legacy = self._script(resolution, "refine", "/tmp/explicit.ckpt", "_2", "--cfg", "job")
            self.assertEqual(legacy.returncode, 0, legacy.stderr)
            self.assertIn("checkpointing.pretrained_model=/tmp/explicit.ckpt\n", legacy.stdout)

    def test_omniscene_defaults_to_one_update_and_allows_legacy_override(self):
        for resolution in ("112x200", "224x400"):
            cfg = OmegaConf.load(ROOT / "config" / "experiment" / f"omniscene_{resolution}.yaml")
            self.assertEqual(cfg.model.encoder.num_refine, 1)
            self.assertEqual(cfg.model.encoder.train_min_refine, 0)
            self.assertEqual(cfg.model.encoder.train_max_refine, 0)
            result = self._script(resolution, "refine", "_legacy", "model.encoder.num_refine=2", "--cfg", "job")
            self.assertEqual(result.returncode, 0, result.stderr)
            args = result.stdout.splitlines()
            self.assertLess(args.index("model.encoder.num_refine=1"), args.index("model.encoder.num_refine=2"))

    def test_existing_run_and_missing_init_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "checkpoint.ckpt").touch()
            result = self._script("112x200", "init", "_1", f"output_dir={tmp}")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Output already exists", result.stderr)
            missing = self._script("112x200", "refine", str(Path(tmp) / "absent.ckpt"), "_1", f"output_dir={tmp}/new")
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("Init checkpoint not found", missing.stderr)
            resume = self._script("112x200", "refine", "_1", f"output_dir={tmp}",
                                  "checkpointing.load=/tmp/resume.ckpt", "checkpointing.pretrained_model=null")
            self.assertEqual(resume.returncode, 0, resume.stderr)


if __name__ == "__main__":
    unittest.main()
