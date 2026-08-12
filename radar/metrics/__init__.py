"""Structure metrics over time: the systemic-risk index and its companions."""

from radar.metrics.absorption import (
    DEFAULT_COMPONENT_FRACTION,
    absorption_ratio,
    effective_dimension,
    eigen_spectrum,
    mean_correlation,
    n_components_for,
    pc1_share,
)
from radar.metrics.rolling import (
    Artifact,
    ArtifactSpec,
    build_artifact,
    correlation_at,
    load_artifact,
)

__all__ = [
    "DEFAULT_COMPONENT_FRACTION",
    "Artifact",
    "ArtifactSpec",
    "absorption_ratio",
    "build_artifact",
    "correlation_at",
    "effective_dimension",
    "eigen_spectrum",
    "load_artifact",
    "mean_correlation",
    "n_components_for",
    "pc1_share",
]
