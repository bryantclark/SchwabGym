"""
SchwabGym Mock Streamer
=======================

Mimics the schwab-py Streamer interface for async data streaming.
"""

import asyncio
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class MockStreamer:
    """
    Mimics schwab.streaming.Streamer.

    Streams Level 1 data from the simulator's PriceEngine.
    """

    def __init__(self, client):
        self.client = client
        self.listening = False
        self.subscriptions: dict[str, list[str]] = {}

    async def start(self, receiver: Callable):
        """
        Mimics schwab-py streamer.start().

        This loop checks if the simulation time has advanced and
        pushes new data to the receiver callback.
        """
        logger.info("MockStreamer started")
        self.listening = True

        last_step = -1

        while self.listening:
            # Check if simulation time advanced
            current_step = self.client.current_step

            if current_step > last_step:
                last_step = current_step

                # Generate messages for active subscriptions
                if "LEVELONE_EQUITIES" in self.subscriptions:
                    symbols = self.subscriptions["LEVELONE_EQUITIES"]
                    quotes = self.client.price_engine.get_quotes_data(symbols)

                    # Transform to stream format
                    # Schwab stream format is usually a list of dicts or a specialized object

                    content = []
                    for sym, data in quotes.items():
                        q = data["quote"]
                        content.append(
                            {
                                "key": sym,
                                "1": q["bidPrice"],
                                "2": q["askPrice"],
                                "3": q["lastPrice"],
                                "4": q["bidSize"],
                                "5": q["askSize"],
                                "8": q["totalVolume"],
                                "9": q["lastSize"],
                                # Add more fields as needed mapping to stream IDs
                            }
                        )

                    msg = {
                        "service": "LEVELONE_EQUITIES",
                        "timestamp": int(
                            self.client.price_engine.get_current_time().timestamp()
                            * 1000
                        ),
                        "content": content,
                    }

                    # Call receiver (awaitable or callable)
                    if asyncio.iscoroutinefunction(receiver):
                        await receiver(msg)
                    else:
                        receiver(msg)

            # Yield control to allow other tasks to run
            await asyncio.sleep(0.001)

    def send(self, requests) -> None:
        """
        Handle subscription requests (ADD, SUBS, UNSUBS).

        Mimics streamer.send(requests).
        """
        # If requests is a list of requests
        if not isinstance(requests, list):
            requests = [requests]

        for req in requests:
            # Duck typing for request objects
            service = getattr(req, "service", None)
            command = getattr(req, "command", None)
            params = getattr(req, "parameters", {})

            if not service or not command:
                logger.warning(f"Invalid stream request: {req}")
                continue

            if command in ("SUBS", "ADD"):
                keys = params.get("keys", "")
                symbols = keys.split(",") if isinstance(keys, str) else keys

                logger.info(f"Streamer subscribing to {service}: {symbols}")
                if command == "ADD" and service in self.subscriptions:
                    existing = set(self.subscriptions[service])
                    existing.update(symbols)
                    self.subscriptions[service] = list(existing)
                else:
                    self.subscriptions[service] = symbols

            elif command in ("UNSUBS", "UNSS"):
                if service in self.subscriptions:
                    del self.subscriptions[service]

    def stop(self) -> None:
        """Stop the streaming loop."""
        self.listening = False
