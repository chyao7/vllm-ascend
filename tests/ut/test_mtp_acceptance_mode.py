#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Tests for MTP acceptance mode support.

Verifies that:
1. MTP method supports all acceptance modes EXCEPT draft_probs
2. EAGLE/EAGLE3 methods continue to support all modes including draft_probs
3. draft_model and other methods do NOT get acceptance mode support
4. collect_draft_probs is False for MTP, True for EAGLE with draft_probs mode
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config


def _make_vllm_config(method, additional_config=None):
    """Create a VllmConfig with a speculative config mock."""
    vllm_config = VllmConfig()
    spec_config = SimpleNamespace(
        method=method,
        disable_padded_drafter_batch=False,
    )
    vllm_config.speculative_config = spec_config
    vllm_config.additional_config = additional_config or {}
    return vllm_config


class TestMTPAcceptanceModeConfig(TestBase):
    """Test that MTP accepts the correct set of acceptance modes."""

    def setUp(self):
        super().setUp()
        clear_ascend_config()

    def tearDown(self):
        clear_ascend_config()

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_legacy_ok(self, _mock):
        cfg = _make_vllm_config("mtp", {"eagle_acceptance_mode": "legacy"})
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "legacy")

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_target_max_ok(self, _mock):
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "target_max",
            "eagle_acceptance_beta": 0.8,
            "eagle_acceptance_target_prob_threshold": 0.1,
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "target_max")
        self.assertEqual(ascend.eagle_acceptance_beta, 0.8)

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_alpha_ok(self, _mock):
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "alpha",
            "eagle_acceptance_alpha": 1.5,
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "alpha")
        self.assertEqual(ascend.eagle_acceptance_alpha, 1.5)

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_entropy_verified_ok(self, _mock):
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "entropy_verified",
            "eagle_acceptance_posterior_threshold": 0.95,
            "eagle_acceptance_posterior_alpha": 0.4,
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "entropy_verified")

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_draft_probs_raises(self, _mock):
        """MTP + draft_probs must raise ValueError."""
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "draft_probs",
        })
        with self.assertRaises(ValueError) as ctx:
            init_ascend_config(cfg)
        self.assertIn("draft_probs", str(ctx.exception))
        self.assertIn("MTP", str(ctx.exception))

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_first_draft_rounds_alpha_fallback_ok(self, _mock):
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "target_max_first_draft_rounds",
            "eagle_acceptance_target_max_draft_rounds": 3,
            "eagle_acceptance_beta": 0.8,
            "eagle_acceptance_target_prob_threshold": 0.1,
            "eagle_acceptance_fallback_mode": "alpha",
            "eagle_acceptance_alpha": 1.5,
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "target_max_first_draft_rounds")
        self.assertEqual(ascend.eagle_acceptance_fallback_mode, "alpha")

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_first_draft_rounds_draft_probs_fallback_raises(self, _mock):
        """MTP + first_draft_rounds + draft_probs fallback must raise."""
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "target_max_first_draft_rounds",
            "eagle_acceptance_target_max_draft_rounds": 3,
            "eagle_acceptance_beta": 0.8,
            "eagle_acceptance_target_prob_threshold": 0.1,
            "eagle_acceptance_fallback_mode": "draft_probs",
        })
        with self.assertRaises(ValueError) as ctx:
            init_ascend_config(cfg)
        self.assertIn("draft_probs", str(ctx.exception))

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_periodic_alpha_secondary_ok(self, _mock):
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "target_max_periodic",
            "eagle_acceptance_secondary_mode": "alpha",
            "eagle_acceptance_target_max_period_steps": 2,
            "eagle_acceptance_secondary_period_steps": 1,
            "eagle_acceptance_beta": 0.8,
            "eagle_acceptance_target_prob_threshold": 0.1,
            "eagle_acceptance_secondary_alpha": 1.5,
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "target_max_periodic")

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_periodic_draft_probs_secondary_raises(self, _mock):
        """MTP + periodic + draft_probs secondary must raise."""
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "target_max_periodic",
            "eagle_acceptance_secondary_mode": "draft_probs",
            "eagle_acceptance_target_max_period_steps": 2,
            "eagle_acceptance_secondary_period_steps": 1,
            "eagle_acceptance_beta": 0.8,
            "eagle_acceptance_target_prob_threshold": 0.1,
        })
        with self.assertRaises(ValueError) as ctx:
            init_ascend_config(cfg)
        self.assertIn("draft_probs", str(ctx.exception))

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mtp_interval_draft_rounds_ok(self, _mock):
        cfg = _make_vllm_config("mtp", {
            "eagle_acceptance_mode": "target_max_interval_draft_rounds",
            "eagle_acceptance_target_max_draft_rounds": 5,
            "eagle_acceptance_target_max_draft_interval": 2,
            "eagle_acceptance_beta": 0.8,
            "eagle_acceptance_target_prob_threshold": 0.1,
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "target_max_interval_draft_rounds")


class TestEAGLEDraftProbsStillWorks(TestBase):
    """Verify EAGLE/EAGLE3 draft_probs is NOT blocked."""

    def setUp(self):
        super().setUp()
        clear_ascend_config()

    def tearDown(self):
        clear_ascend_config()

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_eagle_draft_probs_ok(self, _mock):
        cfg = _make_vllm_config("eagle", {
            "eagle_acceptance_mode": "draft_probs",
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "draft_probs")

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_eagle3_draft_probs_ok(self, _mock):
        cfg = _make_vllm_config("eagle3", {
            "eagle_acceptance_mode": "draft_probs",
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_mode, "draft_probs")

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_eagle_first_draft_rounds_draft_probs_fallback_ok(self, _mock):
        """EAGLE + first_draft_rounds + draft_probs fallback should NOT raise."""
        cfg = _make_vllm_config("eagle", {
            "eagle_acceptance_mode": "target_max_first_draft_rounds",
            "eagle_acceptance_target_max_draft_rounds": 3,
            "eagle_acceptance_beta": 0.8,
            "eagle_acceptance_target_prob_threshold": 0.1,
            "eagle_acceptance_fallback_mode": "draft_probs",
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_fallback_mode, "draft_probs")

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_eagle_periodic_draft_probs_secondary_ok(self, _mock):
        """EAGLE + periodic + draft_probs secondary should NOT raise."""
        cfg = _make_vllm_config("eagle", {
            "eagle_acceptance_mode": "target_max_periodic",
            "eagle_acceptance_secondary_mode": "draft_probs",
            "eagle_acceptance_target_max_period_steps": 2,
            "eagle_acceptance_secondary_period_steps": 1,
            "eagle_acceptance_beta": 0.8,
            "eagle_acceptance_target_prob_threshold": 0.1,
        })
        ascend = init_ascend_config(cfg)
        self.assertEqual(ascend.eagle_acceptance_secondary_mode, "draft_probs")


class TestDraftModelExcluded(TestBase):
    """Verify draft_model and other methods don't get acceptance modes."""

    def setUp(self):
        super().setUp()
        clear_ascend_config()

    def tearDown(self):
        clear_ascend_config()

    @patch("vllm_ascend.ascend_config.logger.warning_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_draft_model_warns(self, _mock, mock_warn):
        """draft_model + acceptance mode should trigger warning."""
        cfg = _make_vllm_config("draft_model", {
            "eagle_acceptance_mode": "target_max",
        })
        init_ascend_config(cfg)
        warned = any("only affects" in str(c) for c in mock_warn.call_args_list)
        self.assertTrue(warned, "Expected warning for draft_model method")

    @patch("vllm_ascend.ascend_config.logger.warning_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_ngram_warns(self, _mock, mock_warn):
        """ngram + acceptance mode should trigger warning."""
        cfg = _make_vllm_config("ngram", {
            "eagle_acceptance_mode": "target_max",
        })
        init_ascend_config(cfg)
        warned = any("only affects" in str(c) for c in mock_warn.call_args_list)
        self.assertTrue(warned, "Expected warning for ngram method")


class TestGetEagleAcceptanceConfig(TestBase):
    """Test _get_eagle_acceptance_config gating function."""

    def setUp(self):
        super().setUp()
        clear_ascend_config()

    def tearDown(self):
        clear_ascend_config()

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def _init_and_get(self, method, _mock):
        from vllm_ascend.sample.rejection_sampler import _get_eagle_acceptance_config

        cfg = _make_vllm_config(method, {
            "eagle_acceptance_mode": "target_max",
            "eagle_acceptance_beta": 0.8,
        })
        init_ascend_config(cfg)
        return _get_eagle_acceptance_config()

    def test_mtp_returns_config(self):
        result = self._init_and_get("mtp")
        self.assertIsNotNone(result)

    def test_eagle_returns_config(self):
        result = self._init_and_get("eagle")
        self.assertIsNotNone(result)

    def test_eagle3_returns_config(self):
        result = self._init_and_get("eagle3")
        self.assertIsNotNone(result)

    def test_draft_model_returns_none(self):
        result = self._init_and_get("draft_model")
        self.assertIsNone(result)

    def test_ngram_returns_none(self):
        result = self._init_and_get("ngram")
        self.assertIsNone(result)

    def test_suffix_returns_none(self):
        result = self._init_and_get("suffix")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
