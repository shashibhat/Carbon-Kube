"""Entrypoint for the carbon poller service."""

import asyncio
import os
import logging

from poller.poller import CarbonPoller, Config

# Set logging level to DEBUG so you see zone and CRD details
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("carbon-kube-poll")


async def main() -> None:
    # -------------------------------
    # Load zones from ENV
    # -------------------------------
    raw_zones = os.getenv("ELECTRICITYMAPS_ZONES", "")
    if raw_zones:
        zones = [z.strip() for z in raw_zones.split(",") if z.strip()]
        LOGGER.debug(f"[ENV] ELECTRICITYMAPS_ZONES -> {zones}")
    else:
        # fallback to old defaults
        zones = ["us-west-2a", "us-west-2b"]
        LOGGER.warning(
            "ELECTRICITYMAPS_ZONES not set — using fallback zones: "
            f"{zones}"
        )

    # -------------------------------
    # Build config
    # -------------------------------
    cfg = Config(
        migration_threshold=200.0,
        zones=zones,
        rl_enabled=True,
    )

    # -------------------------------
    # Start poller
    # -------------------------------
    poller = CarbonPoller(cfg)
    interval = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))

    LOGGER.info("Starting Carbon-Kube poller loop")
    LOGGER.debug(f"Poll interval = {interval}s")
    LOGGER.debug(f"Config Zones = {zones}")

    await poller.run_forever(interval_seconds=interval)


if __name__ == "__main__":
    asyncio.run(main())