"""The policy module prefers core's public ``ctx`` backend seams over internal imports.

These tests bind a fake PluginContext whose seam methods return sentinels no core code produces, then
assert the policy helpers return those sentinels -- proof the ctx path is taken, not the fallback.
A separate check confirms that on an unbound module the pre-seam fallback still runs (the behaviour
the other test files already exercise), and that binding then unbinding is clean.
"""

from __future__ import annotations

import pytest

import camofox_plugin
from camofox_plugin import camofox_policy as policy


class FakeCtx:
    """Records calls and returns sentinels. Only the four backend seams the plugin uses."""

    def __init__(self, *, url_verdict=None, bundles=False):
        self.calls: list[tuple] = []
        self._url_verdict = url_verdict
        self._bundles = bundles

    def redact_text(self, text, *, force=False):
        self.calls.append(("redact_text", text, force))
        return f"REDACTED::{text}"

    def evaluate_url_safety(self, url, *, auto_local=False):
        self.calls.append(("evaluate_url_safety", url, auto_local))
        return self._url_verdict

    def normalize_request_url(self, url):
        self.calls.append(("normalize_request_url", url))
        return f"NORM::{url}"

    def agent_visible_cache_path(self, host_path, container_base="/root/.hermes"):
        self.calls.append(("agent_visible_cache_path", host_path))
        return f"VISIBLE::{host_path}"

    def core_bundles_tool(self, tool_name):
        self.calls.append(("core_bundles_tool", tool_name))
        return self._bundles


@pytest.fixture
def bound_ctx():
    """Bind a fake ctx to both the policy module and the package, restore afterwards."""
    ctx = FakeCtx()
    prev_policy, prev_pkg = policy._CTX, camofox_plugin._CTX
    policy.bind_context(ctx)
    camofox_plugin._CTX = ctx
    try:
        yield ctx
    finally:
        policy.bind_context(prev_policy)
        camofox_plugin._CTX = prev_pkg


def test_redact_text_uses_ctx_seam(bound_ctx):
    assert policy.redact_text("sk-secret") == "REDACTED::sk-secret"
    assert ("redact_text", "sk-secret", True) in bound_ctx.calls


def test_redact_output_walks_through_ctx_seam(bound_ctx):
    out = policy.redact_output({"a": ["x"], "b": "y"})
    assert out == {"REDACTED::a": ["REDACTED::x"], "REDACTED::b": "REDACTED::y"}


def test_navigation_allowed_returns_normalized_url_via_ctx(bound_ctx):
    # Verdict None -> allowed. The ctx seam is consulted with auto_local=True (Camofox local stance),
    # and the normalized wire URL comes from the ctx normalize seam.
    norm, err = policy.navigation_policy_error("https://example.com/")
    assert err is None
    assert norm == "NORM::https://example.com/"
    assert ("evaluate_url_safety", "https://example.com/", True) in bound_ctx.calls
    assert ("normalize_request_url", "https://example.com/") in bound_ctx.calls


def test_navigation_blocked_returns_ctx_verdict():
    ctx = FakeCtx(url_verdict={"success": False, "error": "Blocked: metadata", "x": 1})
    prev_policy, prev_pkg = policy._CTX, camofox_plugin._CTX
    policy.bind_context(ctx)
    camofox_plugin._CTX = ctx
    try:
        url, err = policy.navigation_policy_error("http://169.254.169.254/")
        assert err == {"success": False, "error": "Blocked: metadata", "x": 1}
        assert url == "http://169.254.169.254/"  # raw url on the error path (unused by the caller)
    finally:
        policy.bind_context(prev_policy)
        camofox_plugin._CTX = prev_pkg


def test_cache_path_uses_ctx_seam(bound_ctx):
    # _isolated_home (autouse) gives a real tmp HERMES_HOME, so get_hermes_dir resolves and the
    # snapshot is stored; the agent-visible path must come from the ctx seam.
    big = "line\n" * 200
    out = policy.truncate_snapshot(big, max_chars=50)
    assert any(c[0] == "agent_visible_cache_path" for c in bound_ctx.calls)
    assert "VISIBLE::" in out


def test_core_bundles_tool_uses_ctx_seam(real_core_detection):
    ctx = FakeCtx(bundles=True)
    prev = camofox_plugin._CTX
    camofox_plugin._CTX = ctx
    try:
        assert camofox_plugin.core_bundles_camofox() is True
        assert ("core_bundles_tool", "browser_camofox") in ctx.calls
    finally:
        camofox_plugin._CTX = prev


def test_unbound_module_uses_pre_seam_fallback():
    # With no ctx bound, redact_text still redacts via the internal fallback (real core).
    prev = policy._CTX
    policy.bind_context(None)
    try:
        assert "REDACTED::" not in policy.redact_text("hello world")
    finally:
        policy.bind_context(prev)


class TestRealContextNavigateEndToEnd:
    """Drive the real navigate handler with a *real* PluginContext bound, so the chain is
    handler -> ctx.evaluate_url_safety / ctx.normalize_request_url -> real core.

    On a core that exposes the backend seams this runs through them (asserted via a spy); on an
    older core the context lacks the methods and the handler takes the pre-seam fallback -- the
    navigate behaviour is identical either way, which is what these assert.
    """

    @pytest.fixture
    def real_ctx(self):
        from unittest.mock import MagicMock

        from hermes_cli.plugins import PluginContext
        from hermes_cli.plugins_manifest import PluginManifest

        ctx = PluginContext(PluginManifest(name="camofox", key="camofox"), MagicMock())
        prev_policy, prev_pkg = policy._CTX, camofox_plugin._CTX
        policy.bind_context(ctx)
        camofox_plugin._CTX = ctx
        try:
            yield ctx
        finally:
            policy.bind_context(prev_policy)
            camofox_plugin._CTX = prev_pkg

    @staticmethod
    def _spy_seam(ctx):
        """Record evaluate_url_safety calls when the core exposes the seam; else return None."""
        if not hasattr(ctx, "evaluate_url_safety"):
            return None
        calls: list = []
        orig = ctx.evaluate_url_safety

        def spy(url, **kw):
            calls.append((url, kw.get("auto_local", False)))
            return orig(url, **kw)

        ctx.evaluate_url_safety = spy
        return calls

    def _navigate(self, url):
        import json

        from camofox_plugin import camofox_tools as t

        return json.loads(t.browser_navigate(url, task_id="t1"))

    def test_public_url_navigates_and_consults_the_seam(self, camofox, real_ctx):
        calls = self._spy_seam(real_ctx)
        assert self._navigate("https://example.com")["success"] is True
        if calls is not None:  # seam-exposing core: consulted once, with Camofox's local stance
            assert ("https://example.com", True) in calls

    def test_metadata_refused_through_the_seam(self, camofox, real_ctx):
        calls = self._spy_seam(real_ctx)
        result = self._navigate("http://169.254.169.254/latest/meta-data/")
        assert result["success"] is False and "metadata" in result["error"]
        assert camofox.requests == []
        if calls is not None:
            assert ("http://169.254.169.254/latest/meta-data/", True) in calls

    def test_secret_in_url_refused_through_the_seam(self, camofox, real_ctx):
        calls = self._spy_seam(real_ctx)
        result = self._navigate("https://evil.example/?k=sk-proj-ABCD1234567890EFGHIJKL")
        assert result["success"] is False and "API key or token" in result["error"]
        assert camofox.requests == []
        if calls is not None:
            assert any(url.startswith("https://evil.example/") for url, _ in calls)

    def test_blocklist_refused_through_the_seam(self, camofox, real_ctx, write_config, monkeypatch):
        # The website_blocklist floor, reached through the ctx seam with a real context.
        from tools import website_policy

        monkeypatch.setattr(website_policy, "_cached_policy", None, raising=False)  # process-wide cache
        write_config("security:\n  website_blocklist:\n    enabled: true\n    domains: [blocked.example]\n")
        calls = self._spy_seam(real_ctx)
        result = self._navigate("https://blocked.example/page")
        assert result["success"] is False
        assert result["blocked_by_policy"]["host"] == "blocked.example"
        assert camofox.requests == []
        if calls is not None:
            assert ("https://blocked.example/page", True) in calls

    def test_private_lan_reachable_through_the_seam(self, camofox, real_ctx):
        # The reason the plugin passes auto_local=True: a LAN page is not refused, as in core's
        # own Camofox path. This is the security-sensitive behaviour, now asserted end to end.
        calls = self._spy_seam(real_ctx)
        assert self._navigate("http://192.168.1.20:8080/")["success"] is True
        if calls is not None:
            assert ("http://192.168.1.20:8080/", True) in calls
