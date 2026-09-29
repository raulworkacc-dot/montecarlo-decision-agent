"""Shared fixtures: one reduced pipeline run per test session.

The history keeps its full size (the counterfactual-recovery tests need it), but the
bootstrap and the Monte Carlo are smaller than in production to keep the suite fast.
"""

from __future__ import annotations

import dataclasses

import pytest

from montecarlo_decisions.artifacts import Artifacts, load_artifacts
from montecarlo_decisions.config import ArtifactPaths
from montecarlo_decisions.pipeline import PipelineResult, PreparedCase, prepare_case, run_simulation
from montecarlo_decisions.scenarios import ScenarioConfig, load_config

TEST_SIMULATIONS = 2_000
TEST_BOOTSTRAP_MODELS = 6


@pytest.fixture(scope="session")
def config() -> ScenarioConfig:
    base = load_config()
    return dataclasses.replace(
        base,
        simulation=dataclasses.replace(
            base.simulation, n_bootstrap_models=TEST_BOOTSTRAP_MODELS, chunk_size=500
        ),
    )


@pytest.fixture(scope="session")
def prepared(config: ScenarioConfig) -> PreparedCase:
    return prepare_case(config, n_jobs=2)


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> ArtifactPaths:
    return ArtifactPaths(tmp_path_factory.mktemp("artifacts"))


@pytest.fixture(scope="session")
def result(prepared: PreparedCase, artifact_dir: ArtifactPaths) -> PipelineResult:
    return run_simulation(prepared, artifact_dir, TEST_SIMULATIONS)


@pytest.fixture(scope="session")
def artifacts(result: PipelineResult, artifact_dir: ArtifactPaths) -> Artifacts:
    return load_artifacts(artifact_dir, result.prepared.config)
