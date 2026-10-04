"""Offline checks for the SSE stream parser. No network, no API keys.
Run:  python test_orchestra.py      (silence = pass)
"""
import inspect
import json
import os
import zipfile

import requests

import orchestra_mcp_server as o


def _server_code_without_changelog():
    """Strip module docstring from orchestra_mcp_server.py.

    The changelog names symbols and files this rework deleted, which is exactly
    what a changelog is for. A whole-file substring ban would make honest history
    unwritable, so we scan only executable code. A genuine reintroduction in a
    function body or assignment must still be caught.
    """
    src = open(o.__file__, encoding="utf-8").read()
    parts = src.split('"""', 2)
    assert len(parts) > 2, \
        "could not locate the module docstring in orchestra_mcp_server.py — " \
        "the file's shape changed; update _server_code_without_changelog()"
    return parts[2]


def _sse(obj):
    """One OpenAI-compatible SSE data line, as the wire delivers it."""
    return b"data: " + json.dumps(obj).encode()


def _delta(text, finish=None, model="glm-5.3", provider=None):
    chunk = {"model": model,
             "choices": [{"delta": {"content": text}, "finish_reason": finish}]}
    if provider:
        chunk["provider"] = provider
    return _sse(chunk)


CFG_GLM = {"provider": "glm", "pin": None, "max_out": 131072}
CFG_PINNED = {"provider": "openrouter", "pin": "Stealth", "max_out": 131072}


def test_provenance_checked_on_first_chunk():
    """A wrong-source stream must return [SUBSTITUTED] and leak no content."""
    lines = [_delta("secret", model="some-other-model", provider="Wafer"),
             _delta(" more", model="some-other-model", provider="Wafer")]
    out = o._consume_stream(lines, "stealth/ox-alpha", CFG_PINNED)
    assert out.startswith("[SUBSTITUTED]"), out
    assert "secret" not in out, "content from a substituted host must not leak"


def test_keepalive_comments_are_skipped():
    """OpenRouter sends ': OPENROUTER PROCESSING' comments during long thinking.
    They are not JSON and must not be parsed, but they DO reset the read timeout."""
    lines = [b": OPENROUTER PROCESSING",
             b"",
             _delta("hello ", model="stealth/ox-alpha", provider="Stealth"),
             b": OPENROUTER PROCESSING",
             _delta("world", finish="stop", model="stealth/ox-alpha", provider="Stealth"),
             b"data: [DONE]"]
    out = o._consume_stream(lines, "stealth/ox-alpha", CFG_PINNED)
    assert out == "hello world", repr(out)


def test_silence_returns_partial_content():
    """The whole point of the rework: a stream that dies mid-flight keeps its text."""
    def dying():
        yield _delta("half an ans", model="glm-5.3")
        raise requests.exceptions.ConnectionError("read timed out")

    out = o._consume_stream(dying(), "glm-5.3", CFG_GLM)
    assert out.startswith("[TIMEOUT]"), out
    assert "half an ans" in out, "partial output must survive, not be discarded"


def test_length_finish_returns_truncated_with_content():
    lines = [_delta("as far as I got", finish="length", model="glm-5.3")]
    out = o._consume_stream(lines, "glm-5.3", CFG_GLM)
    assert out.startswith("[TRUNCATED]"), out
    assert "as far as I got" in out


def test_empty_stream_is_tagged_empty():
    lines = [_delta("", finish="stop", model="glm-5.3"), b"data: [DONE]"]
    out = o._consume_stream(lines, "glm-5.3", CFG_GLM)
    assert out.startswith("[EMPTY]"), out


def _thinking(text, finish=None, model="glm-5.3"):
    """A reasoning delta. Measured 2026-09-01: this is what glm-5.3 and deepseek-v4-pro
    actually stream while thinking — `reasoning_content`, never `content`. glm-5.3 sent
    1,281 of these in 25s and produced no content token until 1066s in."""
    return _sse({"model": model,
                 "choices": [{"delta": {"reasoning_content": text},
                              "finish_reason": finish}]})


def test_reasoning_is_preserved_when_no_answer_follows():
    """The waste this rework exists to stop: a model thinks for minutes, the budget runs
    out before an answer starts, and every reasoning token — already billed — is thrown
    away. The trace must come back attached to the [EMPTY] tag."""
    lines = [_thinking("Let G be a group of odd order. "),
             _thinking("By Feit-Thompson we would be done, but reconstructing: "),
             _thinking("consider a minimal counterexample."),
             b"data: [DONE]"]
    out = o._consume_stream(lines, "glm-5.3", CFG_GLM)
    assert out.startswith("[EMPTY]"), out
    assert "minimal counterexample" in out, "reasoning must be preserved, not discarded"
    assert "REASONING" in out, "the caller must be told this is a trace, not an answer"


def _thinking_openrouter(text, finish=None, model="moonshotai/kimi-k3"):
    """OpenRouter's spelling. Measured on Kimi 2026-09-01: OpenRouter normalises to
    `reasoning` (plus a structured `reasoning_details` carrying the same text), while
    the direct APIs send `reasoning_content`. Note `content` is present but EMPTY on
    these frames, which is why a truthiness check on content alone falls through."""
    return _sse({"model": model, "provider": "Moonshot AI",
                 "choices": [{"delta": {"content": "", "role": "assistant",
                                        "reasoning": text,
                                        "reasoning_details": [{"type": "reasoning.text",
                                                               "text": text}]},
                              "finish_reason": finish}]})


def test_both_reasoning_spellings_are_recovered():
    """The first version of this fix handled only `reasoning_content`, so it worked on
    glm-5.3 and deepseek-v4-pro and did NOTHING for kimi-k3, grok-4.6 and
    gemini-3.7-flash — half the fleet, failing silently. Both spellings must work."""
    cfg = {"provider": "openrouter", "pin": None, "order": ["Moonshot AI"],
           "max_out": 128000}
    lines = [_thinking_openrouter("We need to determine whether 2^61-1 "),
             _thinking_openrouter("is prime. Consider Mersenne primes."),
             b"data: [DONE]"]
    out = o._consume_stream(lines, "moonshotai/kimi-k3", cfg)
    assert out.startswith("[EMPTY]"), out
    assert "Mersenne primes" in out, "OpenRouter's `reasoning` spelling must be recovered"


def test_reasoning_details_is_not_double_counted():
    """`reasoning_details` carries the SAME text as `reasoning`, wrapped in typed
    objects. Reading both would duplicate every trace."""
    cfg = {"provider": "openrouter", "pin": None, "order": ["Moonshot AI"],
           "max_out": 128000}
    out = o._consume_stream([_thinking_openrouter("UNIQUE_MARKER "), b"data: [DONE]"],
                            "moonshotai/kimi-k3", cfg)
    assert out.count("UNIQUE_MARKER") == 1, f"duplicated trace: {out!r}"


def test_content_first_stream_does_not_crash():
    """`think` must be bound even when the very first delta carries content, or a
    normal non-reasoning stream raises NameError on its first frame."""
    out = o._consume_stream([_delta("straight answer", finish="stop"), b"data: [DONE]"],
                            "glm-5.3", CFG_GLM)
    assert out == "straight answer", repr(out)


def test_reasoning_counts_toward_progress():
    """Progress was blind during the entire thinking phase — a job working perfectly
    reported ~0 tokens for eighteen minutes. Reasoning deltas must tick the counter."""
    seen = []
    lines = [_thinking(f"step {i} ") for i in range(60)] + [b"data: [DONE]"]
    o._consume_stream(lines, "glm-5.3", CFG_GLM, progress=seen.append)
    assert seen, "progress callback never fired during a 60-chunk reasoning stream"
    assert "glm-5.3" in seen[-1] and "tokens" in seen[-1], seen[-1]


def test_progress_fires_on_a_short_coarse_stream():
    """Delta size varies enormously by model. Kimi streams thousands of token-sized
    deltas; gemini-3.7-flash batches an entire run into 3-7 large ones (measured
    2026-09-01). A progress rule keyed only to `n % 25` never fired for Gemini, so a
    working job looked hung. Any stream that produces output must report at least once."""
    seen = []
    lines = [_thinking("a long batched chunk of reasoning ", model="glm-5.3"),
             _delta("and the answer", finish="stop", model="glm-5.3"),
             b"data: [DONE]"]
    o._consume_stream(lines, "glm-5.3", CFG_GLM, progress=seen.append)
    assert seen, "no progress reported on a 2-delta stream"


def test_answer_wins_over_reasoning_when_both_present():
    """A normal reasoning model run: it thinks, then answers. The answer is the result;
    the trace must NOT be prepended to it."""
    lines = [_thinking("thinking out loud "),
             _delta("The answer is 42.", finish="stop"),
             b"data: [DONE]"]
    out = o._consume_stream(lines, "glm-5.3", CFG_GLM)
    assert out == "The answer is 42.", repr(out)


def test_partial_answer_survives_a_mid_stream_death():
    """When some answer exists, it wins over the trace — the caller wants the partial
    answer, not the working notes that preceded it."""
    def dying():
        yield _thinking("some reasoning ")
        yield _delta("partial ans")
        raise requests.exceptions.ConnectionError("read timed out")

    out = o._consume_stream(dying(), "glm-5.3", CFG_GLM)
    assert out.startswith("[TIMEOUT]"), out
    assert "partial ans" in out
    assert "some reasoning" not in out, "the answer, not the trace, when both exist"


def test_kimi_budget_is_under_its_verified_ceiling():
    """Three different numbers have lived in this field. 1,048,576 is Kimi's CONTEXT
    window and 400s every call; 943,718 is the real max_completion_tokens verified via
    /endpoints on 2026-09-01; what we send must sit under that and leave input room."""
    kimi = o._MODELS["moonshotai/kimi-k3"]["max_out"]
    assert kimi < 943718, f"{kimi} exceeds the verified output ceiling"
    assert kimi > 24000, f"{kimi} is still the pre-rework panic value"


def test_kimi_order_has_no_delisted_hosts():
    """Wafer was delisted; it sat first in the preference list, so every call silently
    fell through to the second host. Same failure mode as stealth/ox-alpha."""
    assert "Wafer" not in o._MODELS["moonshotai/kimi-k3"]["order"]


def test_registry_has_exactly_the_four_fields():
    """Every deleted field must be gone from every entry — a leftover key means a
    call site somewhere is still reading a number this rework removed."""
    allowed = {"provider", "klass", "max_out", "pin", "order"}
    for mid, m in o._MODELS.items():
        extra = set(m) - allowed
        assert not extra, f"{mid} still carries {extra}"
        assert isinstance(m["max_out"], int) and m["max_out"] > 0, \
            f"{mid}: max_out must always be an explicit budget, never None " \
            f"(fleet-card 'Ceiling discipline')"
        assert not (m["pin"] and m.get("order")), f"{mid}: pin XOR order"


def test_effort_never_disables_reasoning():
    """'none' must map to 'low', not to reasoning.enabled=false — that payload is the
    sole cause of the Gemini 3.7 Flash HTTP 400."""
    assert o._effort("none") == "low"
    assert o._effort("max") == "high"
    assert o._effort("garbage") == "high", "unknown effort falls back to the default"


def test_derived_timeout_views_are_gone():
    for dead in ("_DEFAULT_MAX_TOKENS", "_DEFAULT_TIMEOUTS", "_JOB_TIMEOUTS",
                 "_MAXIMUM_YIELD_MAX_TOKENS", "_CALL_RETRY_MULT",
                 "_JOB_BUDGET_SLACK", "_openrouter_reasoning"):
        assert not hasattr(o, dead), f"{dead} should have been deleted"


def test_no_reasoning_chain_is_structurally_gone():
    """The Gemini 400's CAUSE is deleted, so its cure must be too.

    Checked structurally, not by scanning source text: the module changelog names
    what it removed, which is exactly what a changelog is for, and a text ban would
    make honest history unwritable. Task 2's registry and derived-view tests already
    prove default_max / job_timeout are gone by structure.
    """
    params = inspect.signature(o._call).parameters
    for dead in ("_no_reasoning", "_retry", "timeout", "max_tokens"):
        assert dead not in params, f"_call still takes {dead!r}"

    # The sole payload that ever produced Gemini's 400 must be unbuildable.
    for effort in ("none", "low", "medium", "high", "max", "nonsense"):
        assert o._effort(effort) in ("low", "medium", "high"), effort


def test_runners_no_longer_take_timeouts():
    for fn in (o._run_parallel, o._run_adversarial):
        params = inspect.signature(fn).parameters
        assert "timeouts" not in params, f"{fn.__name__} still takes timeouts"
        assert "progress" in params, f"{fn.__name__} must accept progress"


def test_adversarial_still_rejects_same_class_pairs():
    """The cross-class guard is the reason adversarial mode means anything. It must
    survive a refactor that touched every line around it."""
    out = o._run_adversarial("task", model_a="deepseek-v4-flash", model_b="glm-5.3")
    assert out.startswith("[ERROR]"), out
    assert "cross-class" in out.lower(), out


def test_job_wall_budget_is_gone():
    assert not hasattr(o, "_job_wall_budget")

    code_without_docstring = _server_code_without_changelog()
    assert "wall_deadline" not in code_without_docstring
    assert "DeadlineExceeded" not in code_without_docstring


def test_legacy_modes_are_rejected():
    for dead in ("deepseek", "glm"):
        out = o.orchestra_start(mode=dead, prompt="hi")
        assert '"failed"' in out and "unknown mode" in out, out


def test_orchestra_start_signature():
    params = inspect.signature(o.orchestra_start).parameters
    assert "max_tokens" not in params


def test_tools_dropped_the_dead_knobs():
    for fn in (o.call_deepseek, o.call_glm, o.call_model):
        params = inspect.signature(fn).parameters
        assert "max_tokens" not in params, f"{fn.__name__} still takes max_tokens"
        assert "timeout" not in params, f"{fn.__name__} still takes timeout"


def test_list_fleet_reports_only_live_fields():
    rows = json.loads(o.list_fleet())["fleet"]
    assert len(rows) == 6
    for r in rows:
        assert "default_max_tokens" not in r
        assert "timeout_sync_s" not in r and "timeout_job_s" not in r
        assert isinstance(r["max_output_tokens"], int)


def test_docstrings_carry_no_stale_numbers():
    """Both of these were already lying before the rework: call_deepseek claimed a 45s
    flash timeout when the registry said 450, and call_glm claimed max_tokens 16000
    when the registry said 32000. The numbers are gone; the claims must go too."""
    for fn in (o.call_deepseek, o.call_glm):
        doc = fn.__doc__ or ""
        for stale in ("45s", "16000", "24000", "timeout defaults", "max_tokens"):
            assert stale not in doc, f"{fn.__name__} docstring still cites {stale!r}"


def test_glm_reasoning_effort_passthrough():
    """Regression test: GLM must receive reasoning_effort values verbatim (except
    'none' → 'low'), not collapsed through _effort() which maps "max" → "high".
    Without this, every GLM call would silently ask for less depth than it did
    before the refactor, since "max" is the default on every call path."""

    original_post = requests.post
    original_key = os.environ.get("ZHIPU_API_KEY")
    captured = {}

    def fake_post(*args, **kwargs):
        """Capture the payload and raise immediately to prevent any network call."""
        captured['payload'] = kwargs.get('json')
        raise RuntimeError("test capture, not a real error")

    try:
        # _call returns [SKIPPED] (never reaching requests.post) without a key present.
        # Set a dummy explicitly — a real .env next door must not be what makes this pass.
        os.environ["ZHIPU_API_KEY"] = "dummy-test-key"
        # Test "max" passthrough
        requests.post = fake_post
        try:
            o._call("glm-5.3", [{"role": "user", "content": "test"}],
                   glm_reasoning_effort="max")
        except RuntimeError:
            pass  # Expected

        assert captured['payload'] is not None, "payload was not captured"
        assert captured['payload']['reasoning_effort'] == "max", \
            f"'max' should pass through as 'max', got {captured['payload']['reasoning_effort']!r}"

        # Test "none" → "low" remap
        captured.clear()
        try:
            o._call("glm-5.3", [{"role": "user", "content": "test"}],
                   glm_reasoning_effort="none")
        except RuntimeError:
            pass  # Expected

        assert captured['payload'] is not None, "payload was not captured for 'none'"
        assert captured['payload']['reasoning_effort'] == "low", \
            f"'none' should remap to 'low', got {captured['payload']['reasoning_effort']!r}"

        # Test "high" passthrough
        captured.clear()
        try:
            o._call("glm-5.3", [{"role": "user", "content": "test"}],
                   glm_reasoning_effort="high")
        except RuntimeError:
            pass  # Expected

        assert captured['payload']['reasoning_effort'] == "high", \
            f"'high' should pass through as 'high', got {captured['payload']['reasoning_effort']!r}"

    finally:
        requests.post = original_post
        if original_key is None:
            os.environ.pop("ZHIPU_API_KEY", None)
        else:
            os.environ["ZHIPU_API_KEY"] = original_key


def test_dead_registry_file_is_gone():
    here = os.path.dirname(os.path.abspath(o.__file__))
    assert not os.path.exists(os.path.join(here, "fleet_Registry.xml"))
    code_without_docstring = _server_code_without_changelog()
    assert "fleet_Registry" not in code_without_docstring


def test_skill_has_no_dangling_reachability_pointer():
    """fleet-card.md pointed readers at a registry comment that no longer exists."""
    here = os.path.dirname(os.path.abspath(o.__file__))
    with zipfile.ZipFile(os.path.join(here, "orchestra.skill")) as z:
        card = z.read("orchestra/references/fleet-card.md").decode("utf-8")
    assert "reachability math" not in card


def test_ox_alpha_removed_from_models():
    """stealth/ox-alpha was delisted from OpenRouter. _MODELS must have exactly 6 entries,
    with two classes: CN-OW and US-CLOSED. No UNKNOWN class."""
    assert "stealth/ox-alpha" not in o._MODELS, "ox-alpha should be removed from _MODELS"
    assert len(o._MODELS) == 6, f"_MODELS should have 6 entries, got {len(o._MODELS)}"
    classes = set(m["klass"] for m in o._MODELS.values())
    assert classes == {"CN-OW", "US-CLOSED"}, f"Expected {{CN-OW, US-CLOSED}}, got {classes}"


def test_adversarial_default_pair_is_cross_class():
    """_run_adversarial's default model_a and model_b must have different correlation
    classes. The cross-class guard is the entire reason adversarial mode means anything."""
    sig = inspect.signature(o._run_adversarial)
    model_a_default = sig.parameters['model_a'].default
    model_b_default = sig.parameters['model_b'].default
    assert model_a_default in o._MODELS, f"{model_a_default} not in _MODELS"
    assert model_b_default in o._MODELS, f"{model_b_default} not in _MODELS"
    klass_a = o._MODELS[model_a_default]["klass"]
    klass_b = o._MODELS[model_b_default]["klass"]
    assert klass_a != klass_b, \
        f"Default pair {model_a_default} [{klass_a}] and {model_b_default} [{klass_b}] " \
        f"have the same class — cross-class guard broken"


def test_no_model_has_1048576_max_out():
    """1048576 was Kimi K3's original context window. Using it as an output budget leaves
    no room for input. The operator chose 24000 as the budget. No model should revert to 1048576."""
    for model_id, model_cfg in o._MODELS.items():
        assert model_cfg["max_out"] != 1048576, \
            f"{model_id} has max_out=1048576 (context window), not a valid output budget"


def test_http_error_body_is_surfaced():
    """When an HTTP call returns an error, the response body should be included in the
    error message to aid diagnosis. _call() must never raise from error handling."""
    original_post = requests.post
    original_key = os.environ.get("ZHIPU_API_KEY")
    marker_text = "This is a test error body with recognisable marker"

    class FakeResponse:
        def __init__(self):
            self.status_code = 400
            self.text = marker_text + " — some additional details"

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    def fake_post(*args, **kwargs):
        resp = FakeResponse()
        resp.raise_for_status()

    try:
        # _call returns [SKIPPED] (never reaching requests.post) without a key present.
        # Set a dummy explicitly — a real .env next door must not be what makes this pass.
        os.environ["ZHIPU_API_KEY"] = "dummy-test-key"
        requests.post = fake_post
        try:
            result = o._call("glm-5.3", [{"role": "user", "content": "test"}])
        except Exception as e:
            raise AssertionError(f"_call raised {type(e).__name__}: {e} — "
                               "error handling must never raise")

        assert result.startswith("[ERROR]"), \
            f"Expected [ERROR] tag, got {result[:50]!r}"
        assert marker_text in result, \
            f"Error body marker '{marker_text}' not found in result: {result}"

    finally:
        requests.post = original_post
        if original_key is None:
            os.environ.pop("ZHIPU_API_KEY", None)
        else:
            os.environ["ZHIPU_API_KEY"] = original_key


def test_call_streams_with_correct_payload_and_timeout():
    """The v15 rework's central claim, currently untested: _call actually streams.
    Assert the outgoing payload sets stream=True and the registry's max_out budget,
    and that the request timeout is (connect, silence) — a gap-between-chunks bound,
    not a single total-generation one."""
    original_post = requests.post
    original_key = os.environ.get("ZHIPU_API_KEY")
    captured = {}

    def fake_post(*args, **kwargs):
        captured['kwargs'] = kwargs
        raise RuntimeError("test capture, not a real error")

    try:
        # _call returns [SKIPPED] (never reaching requests.post) without a key present.
        # Set a dummy explicitly — a real .env next door must not be what makes this pass.
        os.environ["ZHIPU_API_KEY"] = "dummy-test-key"
        requests.post = fake_post
        try:
            o._call("glm-5.3", [{"role": "user", "content": "test"}])
        except RuntimeError:
            pass  # Expected

        assert captured.get('kwargs') is not None, "requests.post was not called"
        payload = captured['kwargs'].get('json')
        assert payload is not None, "payload was not captured"
        assert payload["stream"] is True, \
            "stream must be True — this is the whole point of the v15 rework"
        assert payload["max_tokens"] == o._MODELS["glm-5.3"]["max_out"]
        assert captured['kwargs'].get('timeout') == (o._CONNECT_SECONDS, o._SILENCE_SECONDS), \
            f"expected timeout=(_CONNECT_SECONDS, _SILENCE_SECONDS), got {captured['kwargs'].get('timeout')!r}"
        # The requests-level flag is the half that makes iter_lines() incremental. Without
        # it the body arrives whole, the read timeout goes back to bounding the entire
        # generation, and a mid-stream death returns "[TIMEOUT] no stream started" with no
        # partial text — silently undoing the rework while the payload assert above stays green.
        assert captured['kwargs'].get('stream') is True, \
            "requests.post must be called with stream=True, not just payload['stream']"
    finally:
        requests.post = original_post
        if original_key is None:
            os.environ.pop("ZHIPU_API_KEY", None)
        else:
            os.environ["ZHIPU_API_KEY"] = original_key


def test_provenance_guard_does_not_disarm_on_bare_role_delta():
    """Regression test for the fail-open bug this review caught: a first frame with
    no model/provider fields (a bare role delta — exactly what OpenAI-compatible
    gateways typically send first) must NOT disarm the provenance guard. A later
    frame declaring a foreign model/provider must still be caught."""
    bare_role_delta = _sse({"choices": [{"delta": {"role": "assistant"}}]})
    foreign_frame = _delta("SECRET", model="some-other-model", provider="Wafer")
    lines = [bare_role_delta, foreign_frame]
    cfg = {"provider": "openrouter", "pin": "xAI", "max_out": 35000}
    out = o._consume_stream(lines, "x-ai/grok-4.6", cfg)
    assert out.startswith("[SUBSTITUTED]"), out
    assert "SECRET" not in out, "content from a substituted host must not leak"


def test_provenance_guard_judges_every_chunk_in_either_field_order():
    """Second regression: the first fix disarmed the guard once a chunk carried EITHER
    model or provider, so whichever field arrived later was never judged. Both orders
    leaked. There is now no disarm flag at all — every chunk is checked.

    The 'model first, wrong host' case is the real one: a pinned OpenRouter model whose
    opening frames carry `model` but not `provider` is exactly the wrong-host scenario
    [SUBSTITUTED] exists to catch."""
    cfg = {"provider": "openrouter", "pin": "xAI", "max_out": 35000}

    provider_first = [_sse({"provider": "xAI", "choices": [{"delta": {"role": "assistant"}}]}),
                      _delta("LEAK", finish="stop", model="llama-3-8b-int4")]
    out = o._consume_stream(provider_first, "x-ai/grok-4.6", cfg)
    assert out.startswith("[SUBSTITUTED]"), out
    assert "LEAK" not in out

    model_first = [_sse({"model": "x-ai/grok-4.6", "choices": [{"delta": {"role": "assistant"}}]}),
                   _sse({"provider": "Wafer",
                         "choices": [{"delta": {"content": "LEAK"}, "finish_reason": "stop"}]})]
    out = o._consume_stream(model_first, "x-ai/grok-4.6", cfg)
    assert out.startswith("[SUBSTITUTED]"), out
    assert "LEAK" not in out


def test_stream_without_provenance_still_returns_content():
    """The guard must not become a false positive. Direct APIs (DeepSeek, Zhipu) never
    send model/provider fields, so a stream where NO chunk carries them is legitimate
    and must return its content rather than tagging or hanging."""
    lines = [_sse({"choices": [{"delta": {"content": "hello "}}]}),
             _sse({"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]})]
    out = o._consume_stream(lines, "glm-5.3", CFG_GLM)
    assert out == "hello world", repr(out)


def test_malformed_data_line_is_skipped():
    """A `data:` line whose payload is not valid JSON must be skipped, not fatal."""
    lines = [b"data: not-json{{{",
             _delta("hello", finish="stop", model="glm-5.3")]
    out = o._consume_stream(lines, "glm-5.3", CFG_GLM)
    assert out == "hello", repr(out)


def test_parallel_rejects_unknown_model():
    """_run_parallel must validate model ids up front, matching _run_adversarial's
    behavior: an unknown model id returns a proper [ERROR] string instead of escaping
    as a bare KeyError, and costs no network call — validation happens before dispatch."""
    original_post = requests.post

    def fail_post(*args, **kwargs):
        raise AssertionError("_run_parallel dispatched before validating model ids")

    try:
        requests.post = fail_post
        out = o._run_parallel(models=("deepseek-v4-flash", "not-a-real-model"))
    finally:
        requests.post = original_post

    assert out.startswith("[ERROR]"), out
    assert "not-a-real-model" in out, out


def test_missing_or_empty_choices_is_skipped():
    """A chunk with no `choices` key at all, or an empty `choices` list, must be
    skipped without failing the stream."""
    lines = [_sse({"model": "glm-5.3"}),               # no choices key at all
             _sse({"model": "glm-5.3", "choices": []}),  # empty choices list
             _delta("hello", finish="stop", model="glm-5.3")]
    out = o._consume_stream(lines, "glm-5.3", CFG_GLM)
    assert out == "hello", repr(out)


def test_registry_matches_live_endpoints():
    """DRIFT DETECTOR — the one test that hits the network.

    Every registry number describing OpenRouter is a hardcoded fact about someone
    else's live system, and each one has already gone stale in this repo:
      - "Wafer" and "stealth/ox-alpha" were delisted; both sat in config for weeks,
        Wafer as FIRST host preference, so calls silently fell through.
      - kimi-k3's max_out held 1,048,576, its CONTEXT window rather than an output
        limit — it 400'd every single call until a live probe caught it.
      - grok-4.6's comment asserted max_completion_tokens was "genuinely null, do not
        guess" long after xAI began declaring 450,000.
    Comments claiming "verified <date>" cannot prevent any of this; only asking the API
    can. This test turns silent drift into a loud failure.

    Skips when OPENROUTER_API_KEY is absent, so the suite still runs fully offline.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return                      # offline run: the other tests still cover logic

    problems = []
    for mid, m in o._MODELS.items():
        if m["provider"] != "openrouter":
            continue                # direct APIs publish no /endpoints to check against
        hosts = m.get("order") or ([m["pin"]] if m.get("pin") else [])
        try:
            resp = requests.get(
                f"https://openrouter.ai/api/v1/models/{mid}/endpoints",
                headers={"Authorization": "Bearer " + key}, timeout=30)
            resp.raise_for_status()
            eps = (resp.json().get("data") or {}).get("endpoints") or []
        except Exception as e:
            problems.append(f"{mid}: could not reach /endpoints ({type(e).__name__})")
            continue

        live = {e.get("provider_name"): e.get("max_completion_tokens") for e in eps}
        if not live:
            problems.append(f"{mid}: /endpoints returned NO hosts — model delisted?")
            continue

        for h in hosts:
            if h not in live:
                problems.append(
                    f"{mid}: configured host {h!r} no longer exists "
                    f"(live hosts: {sorted(k for k in live if k)})")
            elif isinstance(live[h], int) and m["max_out"] > live[h]:
                problems.append(
                    f"{mid}: max_out={m['max_out']} EXCEEDS {h}'s declared "
                    f"max_completion_tokens={live[h]} — this host will reject every call")

    assert not problems, "registry has drifted from live /endpoints:\n  " + \
                         "\n  ".join(problems)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    # stdout is forbidden in the server, but this file is only ever run directly.
    raise SystemExit(0)
