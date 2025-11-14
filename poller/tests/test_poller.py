import asyncio

from poller.poller import CarbonPoller, Config


def test_poll_once_returns_scores():
    cfg = Config(zones=["us-west-2a", "us-west-2b"])
    poller = CarbonPoller(cfg)
    scores = asyncio.run(poller.poll_once())
    assert len(scores) == 2
    zones = {s.zone for s in scores}
    assert zones == {"us-west-2a", "us-west-2b"}
