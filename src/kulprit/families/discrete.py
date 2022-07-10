"""Discrete distribution families."""

import torch
from kulprit.data.data import ModelData
from kulprit.data.submodel import SubModelStructure
from kulprit.families import BaseFamily


class BernoulliFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = False
        self.name = "bernoulli"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic solution to the point-wise dispersion projection."""

        return None


class BinomialFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = False
        self.name = "binomial"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic solution to the point-wise dispersion projection."""

        return None


class CategoricalFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = False
        self.name = "categorical"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic solution to the point-wise dispersion projection."""

        return None


class NegativeBinomialFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = True
        self.name = "negativebinomial"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic solution to the point-wise dispersion projection.

        TODO: Implement analytic dispersion projection for negative binomial family
        """

        raise NotImplementedError
