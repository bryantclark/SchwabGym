"""
SchwabGym Mock Streamer
=======================

Mimics the schwab-py Streamer interface for async data streaming.
"""

import asyncio
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class MockStreamer:
    """
    Mimics schwab.streaming.Streamer.

    Streams Level 1 data from the simulator's PriceEngine.
    """
    def __init__(self, client, queue_size=100):
        self.client = client
        self.listening = False
        self.subscriptions = {} # store active subscriptions

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

                # If we have subscriptions, generate messages
                # For now, we mainly support LEVELONE_EQUITIES
                if "LEVELONE_EQUITIES" in self.subscriptions:
                    symbols = self.subscriptions["LEVELONE_EQUITIES"]
                    quotes = self.client.price_engine.get_quotes_data(symbols)

                    # Transform to stream format
                    # Schwab stream format is usually a list of dicts or a specialized object
                    # We'll send a JSON-like dict as expected by handlers

                    content = []
                    for sym, data in quotes.items():
                        q = data["quote"]
                        content.append({
                            "key": sym,
                            "1": q["bidPrice"],
                            "2": q["askPrice"],
                            "3": q["lastPrice"],
                            "4": q["bidSize"],
                            "5": q["askSize"],
                            "8": q["totalVolume"],
                            "9": q["lastSize"],
                            # Add more fields as needed mapping to stream IDs
                        })

                    msg = {
                        "service": "LEVELONE_EQUITIES",
                        "timestamp": int(self.client.price_engine.get_current_time().timestamp() * 1000),
                        "content": content
                    }

                    # Call receiver (awaitable or callable)
                    if asyncio.iscoroutinefunction(receiver):
                        await receiver(msg)
                    else:
                        receiver(msg)

            # Yield control to allow other tasks to run
            # In a real gym loop, step() is called manually.
            # If the user is running an async loop alongside manual stepping,
            # this sleep allows them to interleave.
            await asyncio.sleep(0.001)

    def send(self, requests):
        """
        Handle subscription requests (ADD, SUBS, UNSS).

        Mimics streamer.send(requests).
        """
        # requests is typically a StreamerRequest object or similar
        # For parity, we assume it has 'service', 'command', 'parameters'
        # But schwab-py might wrap this.
        # If the user passes raw requests:

        # Example request:
        # streamer.send(streamer.level_one_equities("AAPL", "AMD"))

        # We need to handle whatever schwab-py sends.
        # Since we don't import schwab-py, we inspect the object.
        pass # To be implemented if we want full parity, but for now we can just assume basic usage.

        # If requests is a list of requests
        if not isinstance(requests, list):
            requests = [requests]

        for req in requests:
            # simple duck typing
            service = getattr(req, "service", None)
            command = getattr(req, "command", None)
            params = getattr(req, "parameters", {})

            if not service or not command:
                logger.warning(f"Invalid stream request: {req}")
                continue

            if command == "SUBS" or command == "ADD":
                keys = params.get("keys", "")
                fields = params.get("fields", "")

                symbols = keys.split(",") if isinstance(keys, str) else keys

                logger.info(f"Streamer subscribing to {service}: {symbols}")
                self.subscriptions[service] = symbols

            elif command == "UNSS":
                if service in self.subscriptions:
                    del self.subscriptions[service]

    def stop(self):
        self.listening = False
