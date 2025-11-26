import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException


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
        await asyncio.sleep(0.01)
        api_key = os.getenv("ELECTRICITYMAPS_API_KEY", "")
        base_url = os.getenv("ELECTRICITYMAPS_BASE_URL", "https://api.electricitymap.org/v3")
        if api_key:
            try:
                url = f"{base_url}/carbon-intensity/latest?zone={zone}"
                headers = {"auth-token": api_key}
                resp = requests.get(url, headers=headers, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                intensity = float(data.get("carbonIntensity", 0.0))
                return CarbonScore(zone=zone, intensity_g_per_kwh=intensity)
            except Exception:
                pass
        intensity = float(len(zone) * 10)
        return CarbonScore(zone=zone, intensity_g_per_kwh=intensity)

    async def poll_once(self) -> List[CarbonScore]:
        """Fetch carbon scores for all configured zones."""
        tasks = [self.fetch_for_zone(z) for z in self._config.zones]
        return await asyncio.gather(*tasks)

    async def write_crd(self, scores: List[CarbonScore], namespace: str) -> None:
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
        try:
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            api = client.CustomObjectsApi()
            group = "emission.carbon-kube.io"
            version = "v1alpha1"
            plural = "carbonscores"
            name = payload["metadata"]["name"]
            try:
                api.create_namespaced_custom_object(group, version, namespace, plural, payload)
            except ApiException as e:
                if e.status == 409:
                    api.patch_namespaced_custom_object(group, version, namespace, plural, name, payload)
                else:
                    raise
        except Exception as e:
            print("CRD write failed:", e)

    async def run_forever(self, interval_seconds: int = 300, namespace: str = "default") -> None:
        while True:
            scores = await self.poll_once()
            await self.write_crd(scores, namespace)
            await asyncio.sleep(interval_seconds)
