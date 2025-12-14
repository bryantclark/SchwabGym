import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from schwabgym.client import MockClient
from schwabgym.streamer import MockStreamer


@pytest.fixture
def mock_client():
    client = MagicMock(spec=MockClient)
    client.current_step = 0
    client.price_engine = MagicMock()
    client.price_engine.get_current_time.return_value = datetime.datetime(2023, 1, 1)
    client.price_engine.get_quotes_data.return_value = {
        "AAPL": {
            "quote": {
                "symbol": "AAPL",
                "bidPrice": 150.0,
                "askPrice": 150.1,
                "lastPrice": 150.05,
                "bidSize": 100,
                "askSize": 100,
                "totalVolume": 1000,
                "lastSize": 50,
            }
        }
    }
    return client


def test_streamer_lifecycle(mock_client):
    async def _test():
        streamer = MockStreamer(mock_client)

        # Test subscription
        req = MagicMock()
        req.service = "LEVELONE_EQUITIES"
        req.command = "SUBS"
        req.parameters = {"keys": "AAPL", "fields": "0,1,2,3"}

        streamer.send(req)
        assert "LEVELONE_EQUITIES" in streamer.subscriptions
        assert streamer.subscriptions["LEVELONE_EQUITIES"] == ["AAPL"]

        # Test unsubscription
        req_unss = MagicMock()
        req_unss.service = "LEVELONE_EQUITIES"
        req_unss.command = "UNSS"

        streamer.send(req_unss)
        assert "LEVELONE_EQUITIES" not in streamer.subscriptions

    asyncio.run(_test())


def test_streamer_start_loop(mock_client):
    async def _test():
        streamer = MockStreamer(mock_client)

        # Subscribe
        req = MagicMock()
        req.service = "LEVELONE_EQUITIES"
        req.command = "SUBS"
        req.parameters = {"keys": "AAPL"}
        streamer.send(req)

        receiver = AsyncMock()

        # We need to simulate the loop running and client advancing
        # We'll patch asyncio.sleep to stop the loop after a few iterations or raise an exception to break out

        async def side_effect_sleep(delay):
            # Advance client step to trigger data emission
            mock_client.current_step += 1
            if mock_client.current_step > 2:
                streamer.stop()
            return None

        with patch("asyncio.sleep", side_effect=side_effect_sleep):
            await streamer.start(receiver)

        assert receiver.call_count >= 1
        call_args = receiver.call_args[0][0]
        assert call_args["service"] == "LEVELONE_EQUITIES"
        assert call_args["content"][0]["key"] == "AAPL"

    asyncio.run(_test())


def test_streamer_send_list(mock_client):
    streamer = MockStreamer(mock_client)
    req1 = MagicMock()
    req1.service = "LEVELONE_EQUITIES"
    req1.command = "SUBS"
    req1.parameters = {"keys": "AAPL"}

    req2 = MagicMock()
    req2.service = "LEVELONE_EQUITIES"
    req2.command = "ADD"  # ADD should also work like SUBS
    req2.parameters = {"keys": "MSFT"}

    streamer.send([req1, req2])
    assert streamer.subscriptions["LEVELONE_EQUITIES"] == [
        "MSFT"
    ]  # Last one wins in this simple dict impl


def test_streamer_invalid_request(mock_client):
    streamer = MockStreamer(mock_client)
    req = MagicMock()
    del req.service  # Missing service

    streamer.send(req)
    assert len(streamer.subscriptions) == 0
