import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir

from src.misc.wandb_tools import get_wandb_resume_kwargs, update_checkpoint_path


ROOT = Path(__file__).resolve().parents[1]


class TestWandbOffline(unittest.TestCase):
    def test_all_experiments_default_to_offline(self):
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            for experiment in (None, "re10k", "dl3dv", "omniscene_112x200", "omniscene_224x400"):
                with self.subTest(experiment=experiment):
                    overrides = [] if experiment is None else [f"+experiment={experiment}"]
                    cfg = compose(config_name="main", overrides=overrides)
                    self.assertEqual(cfg.wandb.mode, "offline")
            online = compose(config_name="main", overrides=["wandb.mode=online"])
            self.assertEqual(online.wandb.mode, "online")

    def test_offline_resume_is_local_only(self):
        for run_id in (None, "existing-run"):
            with self.subTest(run_id=run_id):
                kwargs = get_wandb_resume_kwargs({"mode": "offline", "id": run_id})
                self.assertIsNone(kwargs["resume"])
                self.assertEqual(kwargs.get("id"), run_id)
        self.assertEqual(get_wandb_resume_kwargs({"mode": "online", "id": None}), {})
        self.assertEqual(
            get_wandb_resume_kwargs({"mode": "online", "id": "existing-run"}),
            {"id": "existing-run", "resume": "must"},
        )

    def test_local_checkpoint_does_not_access_wandb(self):
        with patch("src.misc.wandb_tools.wandb.Api") as api:
            self.assertIsNone(update_checkpoint_path(None, {"mode": "offline"}))
            self.assertEqual(
                update_checkpoint_path("checkpoints/local.ckpt", {"mode": "offline"}),
                Path("checkpoints/local.ckpt"),
            )
            api.assert_not_called()

    def test_offline_remote_checkpoint_fails_before_network_access(self):
        with patch("src.misc.wandb_tools.wandb.Api") as api:
            with self.assertRaisesRegex(ValueError, "local .ckpt path"):
                update_checkpoint_path("wandb://existing-run:v0", {"mode": "offline"})
            api.assert_not_called()

    def test_explicit_online_checkpoint_download_is_preserved(self):
        with patch("src.misc.wandb_tools.download_checkpoint", return_value=Path("model.ckpt")) as download:
            self.assertEqual(
                update_checkpoint_path("wandb://existing-run:v2", {"mode": "online", "project": "resplat"}),
                Path("model.ckpt"),
            )
            download.assert_called_once_with("resplat/existing-run", Path("checkpoints"), "v2")


if __name__ == "__main__":
    unittest.main()
