"""Forward search module."""

from typing import List

import pandas as pd
from kulprit.data.submodel import SubModel
from kulprit.projection.projector import Projector
from kulprit.search import SearchPath


class ForwardSearchPath(SearchPath):
    def __init__(self, projector: Projector) -> None:
        """Initialise search path class."""

        # log the names of the terms in the reference model
        self.ref_terms = projector.model.term_names

        # log the projector object
        self.projector = projector

        # initialise search
        self.k_term_idx = {}
        self.k_term_names = {}
        self.k_submodel = {}
        self.k_dist = {}

    def __repr__(self) -> str:
        """String representation of the search path."""

        path_dict = {
            k: [submodel.structure.term_names, submodel.kl_div]
            for k, submodel in self.k_submodel.items()
        }
        df = pd.DataFrame.from_dict(
            path_dict, orient="index", columns=["Terms", "Distance from reference model"]
        )
        df.index.name = "Model size"
        string = df.to_string()
        return string

    def add_submodel(
        self,
        k: int,
        k_term_names: list,
        k_submodel: SubModel,
        k_dist: float,
    ) -> None:
        """Update search path with new submodel."""

        self.k_term_names[k] = k_term_names
        self.k_submodel[k] = k_submodel
        self.k_dist[k] = k_dist

    def get_candidates(self, k: int) -> List[List]:
        """Method for extracting a list of all candidate submodels.

        Args:
            k (int): The number of terms in the previous submodel, from which we
                wish to find all possible candidate submodels

        Returns:
            List[List]: A list of lists, each containing the terms of all candidate
                submodels
        """

        prev_subset = self.k_term_names[k]
        candidate_additions = list(set(self.ref_terms).difference(prev_subset))
        candidates = [prev_subset + [addition] for addition in candidate_additions]
        return candidates

    def search(self, max_terms: int) -> dict:
        """Forward search through the parameter space.

        Args:
            max_terms (int): Number of terms to perform the forward search
                up to.

        Returns:
            dict: A dictionary of submodels, keyed by the number of terms in the
                submodel.
        """

        # initial intercept-only subset
        k = 0
        k_term_names = ["Intercept"]
        k_submodel = self.projector.project(terms=k_term_names)
        k_dist = k_submodel.kl_div

        # add submodel to search path
        self.add_submodel(
            k=k,
            k_term_names=k_term_names,
            k_submodel=k_submodel,
            k_dist=k_dist,
        )

        # perform forward search through parameter space
        while k < max_terms:
            # get list of candidate submodels, project onto them, and compute
            # their distances
            k_candidates = self.get_candidates(k=k)
            k_projections = [
                self.projector.project(terms=candidate) for candidate in k_candidates
            ]

            # identify the best candidate by distance from reference model
            best_submodel = min(k_projections, key=lambda projection: projection.kl_div)
            best_dist = best_submodel.kl_div

            # retrieve the best candidate's term names and indices
            k_term_names = best_submodel.term_names

            # increment number of parameters
            k += 1

            # add best candidate to search path
            self.add_submodel(
                k=k,
                k_term_names=k_term_names,
                k_submodel=best_submodel,
                k_dist=best_dist,
            )

        # toggle indicator variable and return search path
        self.search_completed = True
        return self.k_submodel
