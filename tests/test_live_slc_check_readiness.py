"""
Tests for live_slc.check_readiness - the Sunday-night read-only Alpaca
account check run before enabling the Scheduled Tasks.
"""
from live_slc.check_readiness import evaluate_readiness


class _FakeAccount:
    def __init__(self, id):
        self.id = id


class _SpyClient:
    """Records every method called. The three read methods return
    configured fixtures; any other method raises - proving
    evaluate_readiness never reaches for a broker mutation, on any path."""

    def __init__(self, *, account_id="acct-1", positions=None, open_orders=None):
        self.calls = []
        self._account_id = account_id
        self._positions = positions if positions is not None else []
        self._open_orders = open_orders if open_orders is not None else []

    def get_account(self):
        self.calls.append("get_account")
        return _FakeAccount(self._account_id)

    def get_all_positions(self):
        self.calls.append("get_all_positions")
        return self._positions

    def get_orders(self, request):
        self.calls.append("get_orders")
        return self._open_orders

    def __getattr__(self, name):
        def _forbidden(*a, **k):
            self.calls.append(name)
            raise AssertionError(f"check_readiness must never call {name}")
        return _forbidden


def test_all_clear_passes_and_calls_only_the_three_read_methods():
    client = _SpyClient(account_id="acct-1")
    result = evaluate_readiness(client, expected_account_id="acct-1")
    assert result.passed is True
    assert result.failures == []
    assert set(client.calls) == {"get_account", "get_all_positions", "get_orders"}


def test_wrong_account_fails_without_touching_env_or_settings():
    """expected_account_id is a plain parameter, never read from
    live_slc.settings inside evaluate_readiness - so this test proves the
    mismatch path with a literal, no .env/settings mutation of any kind."""
    client = _SpyClient(account_id="acct-REAL")
    result = evaluate_readiness(client, expected_account_id="acct-DIFFERENT")
    assert result.passed is False
    assert any("acct-REAL" in f and "acct-DIFFERENT" in f for f in result.failures)


def test_nonzero_positions_fails():
    client = _SpyClient(account_id="acct-1", positions=[object()])
    result = evaluate_readiness(client, expected_account_id="acct-1")
    assert result.passed is False
    assert result.position_count == 1
    assert any("position" in f for f in result.failures)


def test_nonzero_open_orders_fails():
    client = _SpyClient(account_id="acct-1", open_orders=[object()])
    result = evaluate_readiness(client, expected_account_id="acct-1")
    assert result.passed is False
    assert result.open_order_count == 1
    assert any("order" in f for f in result.failures)


def test_only_expected_read_methods_are_ever_called_even_on_failure():
    client = _SpyClient(account_id="acct-WRONG", positions=[object()], open_orders=[object()])
    result = evaluate_readiness(client, expected_account_id="acct-1")
    assert result.passed is False
    assert len(result.failures) == 3
    assert set(client.calls) == {"get_account", "get_all_positions", "get_orders"}
