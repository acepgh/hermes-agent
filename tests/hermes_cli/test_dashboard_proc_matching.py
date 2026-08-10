"""Tests for how ``_scan_dashboard_processes`` decides a process is a stale
long-lived hermes backend worth reaping.

Matching resolves the subcommand as an **argv token**
(``dashboard_procs._hermes_subcommand``) rather than substring-testing the
cmdline.

History: the original implementation substring-matched a fixed pattern list
including ``"hermes_cli.main serve"``.  Profile-scoped gateways spawn as
``python -m hermes_cli.main --profile <name> serve …`` — ``--profile <name>``
sits *between* the entrypoint and the subcommand, so that pattern (and every
other one in the list) failed to match.  Those backends were invisible to the
reaper, so ``hermes update``'s kill-then-respawn cycle degenerated into
respawn-only and leaked one backend — plus its whole MCP subprocess tree —
per update, until the machine ran out of memory.

The token walk deliberately skips option *values*, not just options: ``ps``
joins argv with spaces and applies no quoting, so an unquoted chat prompt
(``hermes chat -q restart the dashboard``) would otherwise present a bare
``dashboard`` token and get the user's chat session killed.
"""

from __future__ import annotations

import pytest

from hermes_cli.dashboard_procs import (
    _hermes_subcommand,
    _matches_reapable_backend,
)


# The exact cmdline shape that leaked 142 backends on 2026-08-07.
LEAKED_PROFILE_SERVE = (
    "/Users/x/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main "
    "--profile alfie serve --host 127.0.0.1 --port 8123"
)


class TestReapableBackends:
    @pytest.mark.parametrize(
        "cmdline",
        [
            pytest.param(LEAKED_PROFILE_SERVE, id="profile-scoped-serve"),
            pytest.param(
                "venv/bin/python -m hermes_cli.main -p coder dashboard --port 9119",
                id="short-p-form",
            ),
            pytest.param(
                "venv/bin/python -m hermes_cli.main --profile=alfie serve",
                id="inline-flag-value",
            ),
            pytest.param("hermes --profile alfie serve", id="console-script-profile"),
            # Shapes the pre-token implementation already handled — these must
            # not regress.
            pytest.param("python -m hermes_cli.main dashboard", id="module-dashboard"),
            pytest.param("python -m hermes_cli.main serve", id="module-serve"),
            pytest.param("/usr/bin/env python3 -m hermes_cli.main serve", id="env-launcher"),
            pytest.param(
                "C:\\x\\venv\\Scripts\\python.exe -m hermes_cli.main serve --host 127.0.0.1",
                id="windows",
            ),
            pytest.param(
                "C:\\x\\venv\\Scripts\\pythonw.exe -m hermes_cli.main --profile p serve",
                id="windows-windowless-profile",
            ),
            pytest.param("hermes dashboard --port 9119", id="console-script"),
            pytest.param("hermes dashboard", id="console-script-bare"),
            pytest.param("hermes serve --port 8300", id="console-script-serve"),
            pytest.param("/usr/local/bin/hermes_cli/main.py serve", id="script-path"),
        ],
    )
    def test_matches(self, cmdline):
        assert _matches_reapable_backend(cmdline) is True

    @pytest.mark.parametrize(
        "cmdline",
        [
            # Killing this is a live outage: the gateway supervisor, not a backend.
            pytest.param(
                "venv/bin/python -m hermes_cli.main --profile alfie gateway run --replace",
                id="live-gateway",
            ),
            # ps does no quoting, so chat prompts arrive as bare words.
            pytest.param("hermes chat -q restart the dashboard", id="chat-prompt"),
            pytest.param(
                "hermes chat -q please serve the dashboard now", id="chat-prompt-both-words"
            ),
            pytest.param("hermes -m gpt-4 chat", id="value-flag-before-subcommand"),
            # `--continue` has nargs="?"; argparse binds the next token as its
            # value, so the walk must too.
            pytest.param("hermes -c serve", id="optional-value-flag"),
            # Merely mentioning hermes must not qualify as an entrypoint.
            pytest.param("grep hermes serve", id="grep"),
            pytest.param("rg --files hermes serve", id="ripgrep"),
            pytest.param("sudo -u bob hermes serve", id="unknown-launcher"),
            pytest.param("tail -f hermes serve.log", id="unrelated-tool"),
            pytest.param("vim hermes_cli/main.py", id="editor"),
            pytest.param("node /some/dashboard/serve.js", id="unrelated-process"),
            pytest.param("hermes update", id="other-subcommand"),
            pytest.param("hermes", id="no-subcommand"),
            pytest.param("", id="empty"),
        ],
    )
    def test_does_not_match(self, cmdline):
        assert _matches_reapable_backend(cmdline) is False


class TestSubcommandResolution:
    """The walk must land on the true positional, not an option's value."""

    @pytest.mark.parametrize(
        "cmdline,expected",
        [
            (LEAKED_PROFILE_SERVE, "serve"),
            ("hermes --profile alfie chat", "chat"),
            ("hermes -m gpt-4 --provider openrouter chat", "chat"),
            ("venv/bin/python -m hermes_cli.main --profile alfie gateway run", "gateway"),
            ("hermes -- serve", "serve"),  # explicit end-of-options
            ("hermes --yolo serve", "serve"),  # store_true consumes nothing
        ],
    )
    def test_resolves_subcommand(self, cmdline, expected):
        assert _hermes_subcommand(cmdline) == expected


def test_profile_flag_is_recognised_as_value_taking():
    """``--profile`` never reaches argparse (main._apply_profile_override
    strips it pre-parse), so it must come from PRE_ARGPARSE_INHERITED_FLAGS.
    If that wiring breaks, ``--profile`` stops consuming its value and the
    walk returns the profile *name* as the subcommand — silently resurrecting
    the original leak."""
    from hermes_cli.dashboard_procs import _global_value_flags

    flags = _global_value_flags()
    assert "--profile" in flags
    assert "-p" in flags
    # The regression itself, stated directly.
    assert _hermes_subcommand("hermes --profile serve serve") == "serve"
