"""Continuous distribution families."""

from kulprit.data.data import ModelData
from kulprit.families import BaseFamily

import numpy as np
import torch


class GaussianFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = True
        self.disp_name_format = "_sigma"
        self.name = "gaussian"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic projection of the Gaussian dispersion parameters.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto

        Returns:
            torch.tensor: The restricted projections of the dispersion parameters
        """

        def _dispersion_proj(
            theta_ast: torch.tensor,
            theta_perp: torch.tensor,
            sigma_ast: torch.tensor,
        ) -> np.ndarray:
            """Analytic solution to the Gaussian dispersion projection.

            We separate this solution from the primary method to allow for
            vectorisation of the projection across samples.

            Args:
                theta_ast (torch.tensor): Reference model posterior parameter
                    sample
                theta_perp (torch.tensor): Submodel projected parameter sample
                sigma_ast (torch.tensor): Reference model posterior dispersion
                    parameter sample

            Returns:
                np.ndarray: The sample projection of the dispersion parameter in
                    a Gaussian model according to the analytic solution
            """

            f = X_ast @ theta_ast
            f_perp = X_perp @ theta_perp
            sigma_perp = torch.sqrt(
                sigma_ast**2
                + 1 / self.data.structure.num_obs * (f - f_perp).T @ (f - f_perp)
            )
            sigma_perp = sigma_perp.numpy()
            return sigma_perp

        # extract parameter draws from both models
        theta_ast = torch.from_numpy(
            self.data.idata.posterior.stack(samples=("chain", "draw"))[
                self.data.structure.term_names
            ]
            .to_array()
            .transpose(*("samples", "variable"))
            .values
        ).float()
        sigma_ast = torch.from_numpy(
            self.data.idata.posterior.stack(samples=("chain", "draw"))[
                self.data.structure.response_name + "_sigma"
            ]
            .transpose()
            .values
        ).float()

        # thin the parameter draws to match the optimisation
        theta_ast = theta_ast[self.data.structure.thinned_idx]
        sigma_ast = sigma_ast[self.data.structure.thinned_idx]

        # extract the design matrix of the reference model
        X_ast = self.data.structure.X

        # project the dispersion parameter
        vec_dispersion_proj = np.vectorize(
            _dispersion_proj,
            signature="(n),(m),()->()",
            doc="Vectorised `_dispersion_proj` method",
        )
        sigma_perp = (
            torch.from_numpy(vec_dispersion_proj(theta_ast, theta_perp, sigma_ast))
            .flatten()
            .float()
        )

        # assure correct shape
        assert sigma_perp.shape == sigma_ast.shape
        return sigma_perp


class PoissonFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = False
        self.disp_name_format = None
        self.name = "poisson"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic solution to the point-wise dispersion projection.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto

        Returns:
            None: Since the Poisson distribution does not admit a dispersion
                parameter
        """

        return None


class BetaFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = False
        self.disp_name_format = None
        self.name = "beta"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic solution to the point-wise dispersion projection.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto

        Returns:
            None: Since the Beta distribution does not admit a dispersion parameter
        """

        return None


class GammaFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = True
        self.disp_name_format = "_alpha"
        self.name = "gamma"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic projection of the Gamma dispersion parameters.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto

        Returns:
            torch.tensor: The restricted projections of the dispersion parameter
        """

        def _dispersion_proj(
            theta_ast: torch.tensor,
            theta_perp: torch.tensor,
        ) -> np.ndarray:
            """Analytic solution to the Gamma dispersion projection.

            We separate this solution from the primary method to allow for
            vectorisation of the projection across samples.

            Args:
                theta_ast (torch.tensor): Reference model posterior parameter
                    sample
                theta_perp (torch.tensor): Submodel projected parameter sample
                sigma_ast (torch.tensor): Reference model posterior dispersion
                    parameter sample

            Returns:
                np.ndarray: The sample projection of the dispersion parameter in
                    a Student-T model according to the analytic solution
            """

            eta = X_ast @ theta_ast
            eta_perp = X_perp @ theta_perp
            f = link(eta)
            dis_perp = torch.sqrt(
                (eta - eta_perp) / (self.data.structure.num_obs * f**2)
            )
            dis_perp = dis_perp.numpy()
            return dis_perp

        # extract parameter draws from both models
        theta_ast = torch.from_numpy(
            self.data.idata.posterior.stack(samples=("chain", "draw"))[
                self.data.structure.term_names
            ]
            .to_array()
            .transpose(*("samples", "variable"))
            .values
        ).float()

        # extract link function from the family
        link = self.data.structure.link.link

        # thin the parameter draws to match the optimisation
        theta_ast = theta_ast[self.data.structure.thinned_idx]

        # extract the design matrix of the reference model
        X_ast = self.data.structure.X

        # project the dispersion parameter
        vec_dispersion_proj = np.vectorize(
            _dispersion_proj,
            signature="(n),(m),()->()",
            doc="Vectorised `_dispersion_proj` method",
        )
        dis_perp = (
            torch.from_numpy(vec_dispersion_proj(theta_ast, theta_perp))
            .flatten()
            .float()
        )

        # assure correct shape
        assert dis_perp.shape == dis_perp.shape
        return dis_perp


class StudentTFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = True
        self.disp_name_format = "_sigma"
        self.name = "t"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic projection of the Student-T dispersion parameters.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto

        Returns:
            torch.tensor: The restricted projections of the dispersion parameters
        """

        def _dispersion_proj(
            theta_ast: torch.tensor,
            theta_perp: torch.tensor,
            sigma_ast: torch.tensor,
        ) -> np.ndarray:
            """Analytic solution to the Student-T dispersion projection.

            We separate this solution from the primary method to allow for
            vectorisation of the projection across samples.

            Args:
                theta_ast (torch.tensor): Reference model posterior parameter
                    sample
                theta_perp (torch.tensor): Submodel projected parameter sample
                sigma_ast (torch.tensor): Reference model posterior dispersion
                    parameter sample

            Returns:
                np.ndarray: The sample projection of the dispersion parameter in
                    a Student-T model according to the analytic solution
            """

            f = X_ast @ theta_ast
            f_perp = X_perp @ theta_perp
            sigma_perp = torch.sqrt(
                1
                / self.data.structure.num_obs
                * (sigma_ast**2 + (f - f_perp).T @ (f - f_perp))
            )
            sigma_perp = sigma_perp.numpy()
            return sigma_perp

        # extract parameter draws from both models
        theta_ast = torch.from_numpy(
            self.data.idata.posterior.stack(samples=("chain", "draw"))[
                self.data.structure.term_names
            ]
            .to_array()
            .transpose(*("samples", "variable"))
            .values
        ).float()
        sigma_ast = torch.from_numpy(
            self.data.idata.posterior.stack(samples=("chain", "draw"))[
                self.data.structure.response_name + "_sigma"
            ]
            .transpose()
            .values
        ).float()

        # thin the parameter draws to match the optimisation
        theta_ast = theta_ast[self.data.structure.thinned_idx]
        sigma_ast = sigma_ast[self.data.structure.thinned_idx]

        # extract the design matrix of the reference model
        X_ast = self.data.structure.X

        # project the dispersion parameter
        vec_dispersion_proj = np.vectorize(
            _dispersion_proj,
            signature="(n),(m),()->()",
            doc="Vectorised `_dispersion_proj` method",
        )
        sigma_perp = (
            torch.from_numpy(vec_dispersion_proj(theta_ast, theta_perp, sigma_ast))
            .flatten()
            .float()
        )

        # assure correct shape
        assert sigma_perp.shape == sigma_ast.shape
        return sigma_perp


class WaldFamily(BaseFamily):
    def __init__(self, data: ModelData) -> None:
        # initialise family object with necessary attributes
        self.data = data
        self.has_dispersion_parameters = True
        self.disp_name_format = "_lam"
        self.name = "wald"

    def solve_dispersion(self, theta_perp: torch.tensor, X_perp: torch.tensor):
        """Analytic solution to the point-wise dispersion projection.

        Args:
            theta_perp (torch.tensor): A PyTorch tensor of the restricted
                parameter draws
            X_perp (np.ndarray): The design matrix of the restricted model we
                are projecting onto
        """

        raise NotImplementedError
