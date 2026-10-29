"""CPU checks for removal of the redundant PCPS budget loss.
Run: python -m unittest discover -s tests -v
"""
import argparse
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from legacy_loss_options import RemovedLossWeight

spec = importlib.util.spec_from_file_location("pcps", ROOT / "code/network/pcps.py")
pcps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcps)
import torch


class BudgetCleanupTests(unittest.TestCase):
    def test_boundary_calibration_and_detached_diagnostic(self):
        model = pcps.PrototypeCalibratedPatchSampling(feature_dim=4, prototype_count=2)
        for n, target, kind in [(1, 1., "high"), (8, 1., "low"),
                                 (256, 1., "mixed"), (4096, 256/4096, "mixed"),
                                 (32, 0., "high"), (32, .5, "low")]:
            with self.subTest(n=n, target=target, kind=kind):
                p = (torch.zeros(n, 1) if kind == "low" else torch.ones(n, 1))
                if kind == "mixed":
                    p = torch.linspace(0., 1., n).reshape(-1, 1)
                p.requires_grad_()
                q = model.calibrate_probability_mean(p, target)
                residual = model.budget_residual(q, target)
                self.assertTrue(torch.isfinite(q).all())
                self.assertFalse(residual.requires_grad)
                self.assertLessEqual(abs(residual.item()), n * model.probability_epsilon + .003)
                q.sum().backward()
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_forward_and_checkpoint_structure(self):
        torch.manual_seed(7)
        model = pcps.PrototypeCalibratedPatchSampling(feature_dim=4, prototype_count=2)
        with torch.no_grad():
            model.prototype_codebook.normal_()
            model.prototype_reputation_logits.add_(torch.randn(2, 1) * .15)
        clone = pcps.PrototypeCalibratedPatchSampling(feature_dim=4, prototype_count=2)
        clone.load_state_dict(model.state_dict(), strict=True)
        logits = torch.randn(16, 1, requires_grad=True)
        out = model(torch.randn(16, 4), logits, target_keep_ratio=.5)
        self.assertIn("budget_residual", out)
        self.assertNotIn("selection_budget_loss", out)
        self.assertFalse(out["budget_residual"].requires_grad)
        (out["prototype_calibration_divergence"] + out["prototype_reputation_loss"]
         + out["codebook_compactness_loss"] + out["codebook_anchor_loss"]).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_legacy_cli_options_in_both_trainers(self):
        for filename in ["train_pcps.py", "train_pcps_survival.py"]:
            tree = ast.parse((ROOT / "code" / filename).read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                     and n.args and isinstance(n.args[0], ast.Constant)
                     and n.args[0].value in ["--selection-budget-weight",
                                            "--reputation-separation-weight", "--reputation-bimodal-weight"]]
            self.assertTrue(calls)
            for call in calls:
                parser = argparse.ArgumentParser()
                exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)], type_ignores=[])),
                             filename, "exec"), {"parser": parser, "RemovedLossWeight": RemovedLossWeight})
                self.assertEqual(vars(parser.parse_args([]))[call.args[0].value[2:].replace("-", "_")], 0.)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    parser.parse_args([call.args[0].value, "0"])
                    self.assertTrue(caught)
                for value in ["1", "-0.1", "nan", "inf"]:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parser.parse_args([call.args[0].value, value])

    def test_no_removed_losses_in_training_objective(self):
        for filename in ["train_pcps.py", "train_pcps_survival.py"]:
            tree = ast.parse((ROOT / "code" / filename).read_text())
            objectives = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "loss" for t in n.targets)]
            self.assertTrue(objectives)
            for objective in objectives:
                text = ast.unparse(objective)
                self.assertNotIn("budget", text)
                self.assertIn("codebook_compactness_loss", text)
                self.assertIn("codebook_anchor_loss", text)


if __name__ == "__main__":
    unittest.main()
