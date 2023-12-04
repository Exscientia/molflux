from typing import Literal, Optional, Type, Union

from numpy.random import RandomState
from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)

try:
    from sklearn.linear_model import LogisticRegression as SKLogisticRegression
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
This is an sklearn logistic regression model.

Logistic regression, despite its name, is a linear model for classification rather
than regression. Logistic regression is also known in the literature as logit
regression, maximum-entropy classification (MaxEnt) or the log-linear classifier.
In this model, the probabilities describing the possible outcomes of a single
trial are modeled using a logistic function.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
    Specify the norm of the penalty:
    - `'none'`: no penalty is added;
    - `'l2'`: add a L2 penalty term and it is the default choice;
    - `'l1'`: add a L1 penalty term;
    - `'elasticnet'`: both L1 and L2 penalty terms are added.
    .. warning::
       Some penalties may not work with some solvers. See the parameter
       `solver` below, to know the compatibility between the penalty and
       solver.
dual : bool, default=False
    Dual or primal formulation. Dual formulation is only implemented for
    l2 penalty with liblinear solver. Prefer dual=False when
    n_samples > n_features.
tol : float, default=1e-4
    Tolerance for stopping criteria.
C : float, default=1.0
    Inverse of regularization strength; must be a positive float.
    Like in support vector machines, smaller values specify stronger
    regularization.
fit_intercept : bool, default=True
    Specifies if a constant (a.k.a. bias or intercept) should be
    added to the decision function.
intercept_scaling : float, default=1
    Useful only when the solver 'liblinear' is used
    and self.fit_intercept is set to True. In this case, x becomes
    [x, self.intercept_scaling],
    i.e. a "synthetic" feature with constant value equal to
    intercept_scaling is appended to the instance vector.
    The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
    Note! the synthetic feature weight is subject to l1/l2 regularization
    as all other features.
    To lessen the effect of regularization on synthetic feature weight
    (and therefore on the intercept) intercept_scaling has to be increased.
class_weight : dict or 'balanced', default=None
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one.
    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``.
    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if sample_weight is specified.
random_state : int, RandomState instance, default=None
    Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
    data.
solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
        default='lbfgs'
    Algorithm to use in the optimization problem. Default is 'lbfgs'.
    To choose a solver, you might want to consider the following aspects:
        - For small datasets, 'liblinear' is a good choice, whereas 'sag'
          and 'saga' are faster for large ones;
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
          'lbfgs' handle multinomial loss;
        - 'liblinear' is limited to one-versus-rest schemes.
    .. warning::
       The choice of the algorithm depends on the penalty chosen:
       Supported penalties by solver:
       - 'newton-cg'   -   ['l2', 'none']
       - 'lbfgs'       -   ['l2', 'none']
       - 'liblinear'   -   ['l1', 'l2']
       - 'sag'         -   ['l2', 'none']
       - 'saga'        -   ['elasticnet', 'l1', 'l2', 'none']
    .. note::
       'sag' and 'saga' fast convergence is only guaranteed on
       features with approximately the same scale. You can
       preprocess the data with a scaler from :mod:`sklearn.preprocessing`.
    .. seealso::
       Refer to the User Guide for more information regarding
       :class:`LogisticRegression` and more specifically the
       `Table <https://scikit-learn.org/dev/modules/linear_model.html#logistic-regression>`_
       summarazing solver/penalty supports.
max_iter : int, default=100
    Maximum number of iterations taken for the solvers to converge.
multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
    If the option chosen is 'ovr', then a binary problem is fit for each
    label. For 'multinomial' the loss minimised is the multinomial loss fit
    across the entire probability distribution, *even when the data is
    binary*. 'multinomial' is unavailable when solver='liblinear'.
    'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
    and otherwise selects 'multinomial'.
verbose : int, default=0
    For the liblinear and lbfgs solvers set verbose to any positive
    number for verbosity.
warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    Useless for liblinear solver.
n_jobs : int, default=None
    Number of CPU cores used when parallelizing over classes if
    multi_class='ovr'". This parameter is ignored when the ``solver`` is
    set to 'liblinear' regardless of whether 'multi_class' is specified or
    not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
    context. ``-1`` means using all processors.
l1_ratio : float, default=None
    The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
    used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
    to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
    to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
    combination of L1 and L2.
"""
Penalty = Literal["l1", "l2", "elasticnet", "none"]
ClassWeight = Optional[Union[dict, Literal["balanced"]]]
Solver = Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
MultiClass = Literal["auto", "ovr", "multinomial"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class LogisticRegressorConfig(ModelConfig):
    penalty: Penalty = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = 1
    fit_intercept: bool = True
    intercept_scaling: float = 1
    class_weight: ClassWeight = None
    random_state: Optional[Union[int, RandomState]] = None
    solver: Solver = "lbfgs"
    max_iter: int = 100
    multi_class: MultiClass = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: Optional[int] = None
    l1_ratio: Optional[float] = None


class LogisticRegressor(
    SKLearnClassificationMixin,
    SKLearnModelBase[LogisticRegressorConfig],
):
    @property
    def _config_builder(self) -> Type[LogisticRegressorConfig]:
        return LogisticRegressorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKLogisticRegression:
        config = self.model_config
        return SKLogisticRegression(
            penalty=config.penalty,
            dual=config.dual,
            tol=config.tol,
            C=config.C,
            fit_intercept=config.fit_intercept,
            intercept_scaling=config.intercept_scaling,
            class_weight=config.class_weight,
            random_state=config.random_state,
            solver=config.solver,
            max_iter=config.max_iter,
            multi_class=config.multi_class,
            verbose=config.verbose,
            warm_start=config.warm_start,
            n_jobs=config.n_jobs,
            l1_ratio=config.l1_ratio,
        )
