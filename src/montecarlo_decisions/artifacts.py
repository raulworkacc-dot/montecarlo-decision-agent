"""Read-only access to the outputs of a pipeline run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from montecarlo_decisions.config import ArtifactPaths
from montecarlo_decisions.scenarios import ScenarioConfig, load_config


class ArtifactsMissingError(FileNotFoundError):
    """The pipeline has not been run for this artifact directory."""


@dataclass(frozen=True)
class Artifacts:
    paths: ArtifactPaths
    config: ScenarioConfig
    dataset: pd.DataFrame
    uplift: pd.DataFrame
    ads_regimes: pd.DataFrame
    simulations: pd.DataFrame
    summary: pd.DataFrame
    model_report: dict
    checks: list[dict]

    @property
    def fingerprint(self) -> str:
        """Identifies the simulation results a memo was written for."""
        return summary_fingerprint(self.paths.summary)


def summary_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_artifacts(paths: ArtifactPaths, config: ScenarioConfig | None = None) -> Artifacts:
    missing = [path.name for path in paths.core_outputs() if not path.is_file()]
    if missing:
        raise ArtifactsMissingError(
            f"Missing artifacts in {paths.root}: {', '.join(missing)}. Run `mcd pipeline` first."
        )
    return Artifacts(
        paths=paths,
        config=config or load_config(),
        dataset=pd.read_csv(paths.dataset),
        uplift=pd.read_csv(paths.uplift),
        ads_regimes=pd.read_csv(paths.ads_regimes),
        simulations=pd.read_csv(paths.simulations),
        summary=pd.read_csv(paths.summary),
        model_report=json.loads(paths.model_report.read_text(encoding="utf-8")),
        checks=json.loads(paths.checks.read_text(encoding="utf-8")),
    )
