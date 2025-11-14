"""Entrypoint for the carbon poller service."""

import asyncio
import logging

from poller.poller import CarbonPoller, Config

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("carbon-kube-poll")


async def main() -> None:
    cfg = Config(
        migration_threshold=200.0,
        zones=["us-west-2a", "us-west-2b"],
        rl_enabled=True,
    )
    poller = CarbonPoller(cfg)
    LOGGER.info("Starting Carbon-Kube poller loop")
    await poller.run_forever(interval_seconds=300)


if __name__ == "__main__":
    asyncio.run(main())
