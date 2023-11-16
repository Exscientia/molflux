# type: ignore
from typing import Optional

import numpy as np
from scipy.sparse import issparse
from scipy.special import logsumexp
from scipy.stats import entropy
from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.utils.extmath import safe_sparse_dot


class CorrectedNB(_BaseDiscreteNB):
    """
    Class to implement a Naive Bayes classifier with corrected probabilities
    based on the paper https://pubs.acs.org/doi/10.1021/jm0303195.

    Parameters
    ----------
    alpha : float or array-like of shape (n_features,), default=1.0
        Additive (Laplace/Lidstone) smoothing parameter
        (set alpha=0 for no smoothing).

    fit_prior : bool, default=True
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None) -> None:
        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            force_alpha=True,
        )

    def _update_feature_log_prob(self, alpha: float = 1) -> None:
        """
        Calculates normalised Laplacian smoothed corrected log probabilties for
        each feature as shown in paper: https://pubs.acs.org/doi/10.1021/jm0303195.
        Concretely:

            P_corr = ((F_c + alpha)/(F_tot + alpha/(C_c/C_tot)))/(C_c/C_tot)

        where F_c is the feature frequency of feature F for class c, alpha is
        a smoothing factor (=1 by default), F_tot is the total feature
        frequency of feature F, C_c is the class count of class c, C_tot is the
        total number of samples (sum of class counts for all classes).

        Parameters
        ----------
        alpha : float = 1, optional
            The smoothing parameter to use, default is 1. Called internally
            during fitting with self.alpha.
        """
        total_samples = self.class_count_.sum()
        numerator = self.feature_count_ + alpha
        pbases = self.class_count_ / total_samples
        denominator = self.feature_count_.sum(axis=0) + (
            alpha / (pbases).reshape(-1, 1)
        )
        p_smooth = numerator / denominator
        self.feature_log_prob_ = np.log(p_smooth / pbases.reshape(-1, 1))

    @property
    def feature_entropy_(self) -> np.ndarray:
        """
        Feature entropy property. Obtained by taking the entropy of
        each features log probability for each class.

        Returns
        -------
        np.ndarray :
            The entropy of each feature, given by the feature log
            probability for each class.
        """
        return entropy(self.feature_prob_, axis=0)

    @property
    def presence_feature_entropy(self) -> np.ndarray:
        """
        Returns the entropy component for the presence of each feature in the
        dataset.
        """
        probs = np.sum(self.feature_count_, axis=1) / np.sum(self.class_count_)
        return probs * np.log(probs)

    @property
    def absence_feature_entropy(self) -> np.ndarray:
        """
        Returns the entropy component for the absence of each feature in the
        dataset.
        """
        probs = np.sum(self.class_count_) - np.sum(
            self.feature_count_,
            axis=1,
        ) / np.sum(self.class_count_)
        return probs * np.log(probs)

    @property
    def feature_prob_(self) -> np.ndarray:
        """
        Property to obtain feature probability. Exponential of
        feature log prob.

        Returns
        -------
        np.ndarray :
            The probability of the features for each class.
        """
        return np.exp(self.feature_log_prob_)

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the log likelihood for each sample in X for each class.
        Concretely, takes the dot product between X and feature log prob.

        Parameters
        ----------
        X : np.ndarray
            The binary feature data to obtain the log likelihoods
            for.

        Returns
        -------
        np.ndarray :
            The log likelihoods for each sample in X for each class.
        """
        return safe_sparse_dot(X, self.feature_log_prob_.T)

    def _joint_log_likelihood(self, X):
        return self.joint_log_likelihood(X)

    def joint_log_likelihood(self, X: np.ndarray):
        """
        Function to obtain the joint log likelihood. Equivalent to
        log(P(class)*product(P(feature_i|class))))
        Parameters
        ----------
        X : np.ndarray
            The data to obtain the joint log likelihood from.

        Returns
        -------
        np.ndarray :
            The joint log likelihood for each sample in X for each
            class.
        """
        jll = self.log_likelihood(X)
        # jll += self.class_log_prior_
        return jll

    def predict_log_proba(self, X: np.ndarray):
        """
        Function to get the log probability of each class for each
        sample in X.

        Parameters
        ----------
        X : np.ndarray
            The X data to obtain the log probabilities for.

        Returns
        -------
        np.ndarray :
            An array of shape (n_samples, n_classes) for each sample in
            X. Each sample has the log probabilities for each class.
        """
        jll = self.joint_log_likelihood(X)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Same as predict_log_proba but simply the exponential of the
        values.

        Parameters
        ----------
        X : np.ndarray
            The X data to obtain the probabilities for.

        Returns
        -------
        np.ndarray :
            An array of shape (n_samples, n_classes) for each sample in
            X. Each sample has the probabilities for each class.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Function to predict labels for each sample in X. Uses the MAE
        decision rule, i.e. class with highest probability -> prediction.

        Parameters
        ----------
        X : np.ndarray
            X data to classify, must be binary.

        Returns
        -------
        np.ndarray :
            The labels from .classes_ for each sample in X.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def entropy(self, X: np.ndarray) -> np.ndarray:
        """
        The entropy of each sample in X. Calculated by taking the
        entropy of each class probability for each sample in X.

        Parameters
        ----------
        X : np.ndarray
            The samples to obtain the entropy for. Must be binary.

        Returns
        -------
        np.ndarray :
            The entropy for each sample in X.
        """
        entropy_vals = entropy(self.predict_proba(X), axis=1)
        return entropy_vals

    def empirical_log_likelihood(self, type_="fi|c") -> np.ndarray:
        """
        Obtains the empirical log likelihoods.

        Parameters
        ----------
        type_ : str, optional {'fi|c', 'c|fi'}
            String denoting the likelihood probabilities returned.
            Options are 'fi|c' for probability of feature i given class
            c, or 'c|fi' for probability of class c given feature i.

        Returns
        -------
        np.ndarray :
            The log likelihood for either p(fi|c) or p(c|fi).
        """
        if type_ == "fi|c":
            llh = self.feature_count_ / self.class_count_.reshape(-1, 1)
        elif type_ == "c|fi":
            llh = self.feature_count_ / self.feature_count_.sum(axis=0)
        else:
            raise NotImplementedError("Likelihood type must be 'fi|c' or 'c|fi'.")
        return np.log(llh)

    def empirical_feature_entropy(self, type_: str = "fi|c") -> np.ndarray:
        """
        Calculates the empirical feature entropy by a given conditional `type_`.

        Parameters
        ----------
        type_ : str = 'fi|c', {'fi|c', 'c|fi'}
            The conditional type for the feature entropy, either feature i
            given the class `c` or class c given feature i.

        Returns
        -------
        np.ndarray :
            The empirical feature entropy.
        """
        lh = np.exp(self.empirical_log_likelihood(type_=type_))
        return entropy(lh, axis=0)

    def empirical_feature_entropy_neg_class(self, class_: int = 1, type_: str = "fi|c"):
        """
        Calculates the empirical feature entropy for a class compared to against
        the negative of the class.

        Parameters
        ----------
        class_ : int = 1, optional
            The class to use for the feature negative entropy. Default is `1`,
            i.e. class self.classes_[class_] will be used.
        type_ : str = 'fi|c', {'fi|c', 'c|fi'}
            The conditional type for the feature entropy, either feature i
            given the class `c` or class c given feature i.

        Returns
        -------
        np.ndarray :
            The empirical feature entropy for a class against 1 - the probability
            of that class.
        """
        lh = np.exp(self.empirical_log_likelihood(type_=type_))[class_]
        pk = np.array([lh, 1 - lh])
        return entropy(pk, axis=0)

    def _count(self, X, y):
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        self.feature_count_ += safe_sparse_dot(y.T, X)
        self.class_count_ += np.sum(y, axis=0)


class CoverageNB(CorrectedNB):
    """
    Classifier similar to `CorrectedNB` but with additional methods relevant
    to Coverage Score algorithm - such as feature coverage and feature final
    coverage.
    """

    @property
    def feature_coverage_(self) -> np.ndarray:
        """
        Property indicating the feature 'coverage'. This is calculated
        as -(ln(P(C_c|feature_i) - ln(P(C_c))) for each class c.

        Returns
        -------
        np.ndarray :
            The feature 'coverage' for each class k for each feature.
        """
        cov = -self.feature_log_prob_
        return cov

    @property
    def feature_coverage_entropy_(self) -> np.ndarray:
        """
        Returns the feature coverage multiplied by the feature entropy.

        Returns
        -------
        np.ndarray :
            The feature coverage multiplied by the feature entropy.
        """
        raw_cov = self.feature_coverage_
        shan_ents = self.get_feature_entropy_class(norm=True)
        return raw_cov * shan_ents

    @property
    def feature_final_coverage_(self) -> np.ndarray:
        """
        Property indicating the 'final coverage' of a feature.

        Returns
        -------
        np.ndarray :
            The final coverage for each feature for each class.
        """
        # Get raw coverage
        raw_cov = self.feature_coverage_
        shan_ents = self.get_feature_entropy_class(norm=True)
        # Coverage is base_cov * entropy.
        final_cov = self.feature_coverage_entropy_
        frac_count = self.feature_count_ / self.class_count_.reshape(-1, 1)
        # For frequencies > half the samples, alter coverage score (2 - entropy)
        final_cov[(frac_count > 0.5) & (raw_cov < 0)] = raw_cov[
            (frac_count > 0.5) & (raw_cov < 0)
        ] * (2 - shan_ents[(frac_count > 0.5) & (raw_cov < 0)])
        return final_cov

    def get_feature_entropy_class(
        self,
        norm: bool = True,
    ) -> np.ndarray:
        """
        Function to obtain the entropy for each feature for each class.
        Calculated as H(F_c) = -(P(C_c)*log(P(C_c)) + (1-P(C_c))*log((1-P(C_c)))
        for each class c.

        Parameters
        ----------
        norm : bool, optional
            Determines whether to normalize the entropies such that
            they fall between 0 and 1. Default is True.

        Returns
        -------
        np.ndarray :
            Returns the entropy for each feature, for each class c.
        """
        probs = np.exp(self.empirical_log_likelihood(type_="fi|c"))
        denom = np.log(2) if norm else 1
        neg_probs = 1 - probs

        return entropy([probs, neg_probs], axis=0) / denom


class PipelinePilotNB(CorrectedNB):
    """
    Naive Bayes classifier explicitly used in PipelinePilot. Main difference
    is the determination of the base probabilities for calculating a smoothed
    normalised probability.
    """

    def _update_feature_log_prob(self, alpha: Optional[float] = None) -> None:
        """
        Updating the feature log probability to use a normalised, Laplacian
        corrected feature probability as used by PipelinePilot.
        Concretely:

            P_corr = ((F_c + alpha)/(F_tot + alpha/(FC_c/FC_tot)))/(FC_c/FC_tot)

        where F_c is the feature frequency of feature F for class c, alpha is
        a smoothing factor (=1 by default), F_tot is the total feature
        frequency of feature F, FC_c is the **total feature** count of class c,
        FC_tot is the **total feature count of all samples**
        (sum of feature counts for all classes).

        Parameters
        ----------
        alpha : float = 1, optional
            The smoothing parameter to use, default is 1. Called internally
            during fitting with self.alpha.
        """
        total_feature_counts = self.feature_count_.sum()
        summed_feature_counts = self.feature_count_.sum(axis=1)
        feature_count_tot = self.feature_count_.sum(axis=0)

        pbases = summed_feature_counts / total_feature_counts

        numerator = self.feature_count_ + alpha
        denominator = feature_count_tot + 1 / pbases.reshape(-1, 1)

        p_smooth = numerator / denominator

        self.feature_log_prob_ = np.log(p_smooth / pbases.reshape(-1, 1))
