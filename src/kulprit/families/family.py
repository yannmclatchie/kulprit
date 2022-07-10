"""Core family class to be accessed"""

import torch

from kulprit.data.data import ModelData
from kulprit.families import BaseFamily
from kulprit.families.continuous import (
    BetaFamily,
    GammaFamily,
    GaussianFamily,
    PoissonFamily,
    StudentTFamily,
    WaldFamily,
)
from kulprit.families.discrete import (
    BernoulliFamily,
    BinomialFamily,
    CategoricalFamily,
    NegativeBinomialFamily,
)


class Family:
    def __init__(self, data: ModelData) -> None:
        """Family factory constructor.

        Args:
            data (kulprit.data.ModelData): Reference model dataclass object
        """

        # log model data and family name
        self.data = data
        self.family_name = data.structure.family

        # define all available family classes
        self.family_dict = {
            "bernoulli": BernoulliFamily,
            "beta": BetaFamily,
            "binomial": BinomialFamily,
            "categorical": CategoricalFamily,
            "gamma": GammaFamily,
            "gaussian": GaussianFamily,
            "negativebinomial": NegativeBinomialFamily,
            "poisson": PoissonFamily,
            "t": StudentTFamily,
            "wald": WaldFamily,
        }

        # test family name
        if self.family_name not in self.family_dict.keys():
            raise NotImplementedError(
                f"The {self.family_name} family has not yet been implemented."
            )

        # build BaseFamily object
        self.family = self.factory_method()

    def factory_method(self) -> BaseFamily:
        """Choose the appropriate family class given the model."""

        # return appropriate family class given model variate family
        family_class = self.family_dict[self.family_name]
        return family_class(self.data)

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic projection of the model dispersion parameters.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto

        Returns:
            torch.tensor: The restricted projections of the dispersion parameters
        """

        # test whether or not the family has dispersion parameters
        if not self.family.has_dispersion_parameters:  # pragma: no cover
            return None

        # compute the solution and return
        solution = self.family.solve_dispersion(theta_perp=theta_perp, X_perp=X_perp)
        return solution
