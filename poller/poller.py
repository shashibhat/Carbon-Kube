import asyncio
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CarbonScore:
    """In-memory carbon score representation."""

    zone: str
    intensity_g_per_kwh: float
    cpu_multiplier: float = 1.0


@dataclass
class Config:
    """Runtime configuration for the poller service."""

    migration_threshold: float = 200.0
    zones: List[str] | None = None
    rl_enabled: bool = False

    def __post_init__(self) -> None:
        if self.zones is None:
            self.zones = []


class CarbonPoller:
    """Async poller that fetches scores and writes them to a CRD.

    In this reference implementation, the external API calls are mocked,
    and the CRD write is printed to stdout. You can replace these parts
    with real integrations (Electricity Maps, kubernetes_asyncio, etc.).
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    async def fetch_for_zone(self, zone: str) -> CarbonScore:
        """Simulate an external API call for the given zone."""
        await asyncio.sleep(0.01)
        # Deterministic fake intensity: proportional to zone name length.
        intensity = float(len(zone) * 10)
        return CarbonScore(zone=zone, intensity_g_per_kwh=intensity)

    async def poll_once(self) -> List[CarbonScore]:
        """Fetch carbon scores for all configured zones."""
        tasks = [self.fetch_for_zone(z) for z in self._config.zones]
        return await asyncio.gather(*tasks)

    async def write_crd(self, scores: List[CarbonScore]) -> None:
        """Write the scores to a CarbonScore CRD (simulated).

        Replace this with a real call to kubernetes_asyncio.CustomObjectsApi
        in your cluster.
        """
        payload: Dict[str, object] = {
            "apiVersion": "emission.carbon-kube.io/v1alpha1",
            "kind": "CarbonScore",
            "metadata": {"name": "global"},
            "spec": {
                "scores": [
                    {
                        "zone": s.zone,
                        "intensity_g_per_kwh": s.intensity_g_per_kwh,
                        "cpu_multiplier": s.cpu_multiplier,
                    }
                    for s in scores
                ]
            },
        }
        print("Would write CRD payload:", payload)

    async def run_forever(self, interval_seconds: int = 300) -> None:
        """Main polling loop."""
        while True:
            scores = await self.poll_once()
            await self.write_crd(scores)
            await asyncio.sleep(interval_seconds)
