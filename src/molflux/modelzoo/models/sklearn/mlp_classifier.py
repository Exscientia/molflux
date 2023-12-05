from dataclasses import field
from typing import List, Literal, Type, Union

from numpy.random import RandomState
from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)

try:
    from sklearn.neural_network import MLPClassifier as SKMLPClassifier
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
Multi-layer Perceptron classifier.

This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

Notes
-----
MLPClassifier trains iteratively since at each time step
the partial derivatives of the loss function with respect to the model
parameters are computed to update the parameters.
It can also have a regularization term added to the loss function
that shrinks model parameters to prevent overfitting.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
    The ith molflux represents the number of neurons in the ith
    hidden layer.
activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
    Activation function for the hidden layer.
    - 'identity', no-op activation, useful to implement linear bottleneck,
      returns f(x) = x
    - 'logistic', the logistic sigmoid function,
      returns f(x) = 1 / (1 + exp(-x)).
    - 'tanh', the hyperbolic tan function,
      returns f(x) = tanh(x).
    - 'relu', the rectified linear unit function,
      returns f(x) = max(0, x)
solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
    The solver for weight optimization.
    - 'lbfgs' is an optimizer in the family of quasi-Newton methods.
    - 'sgd' refers to stochastic gradient descent.
    - 'adam' refers to a stochastic gradient-based optimizer proposed
      by Kingma, Diederik, and Jimmy Ba
    Note: The default solver 'adam' works pretty well on relatively
    large datasets (with thousands of training samples or more) in terms of
    both training time and validation score.
    For small datasets, however, 'lbfgs' can converge faster and perform
    better.
alpha : float, default=0.0001
    Strength of the L2 regularization term. The L2 regularization term
    is divided by the sample size when added to the loss.
batch_size : int, default='auto'
    Size of minibatches for stochastic optimizers.
    If the solver is 'lbfgs', the classifier will not use minibatch.
    When set to "auto", `batch_size=min(200, n_samples)`.
learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
    Learning rate schedule for weight updates.
    - 'constant' is a constant learning rate given by
      'learning_rate_init'.
    - 'invscaling' gradually decreases the learning rate at each
      time step 't' using an inverse scaling exponent of 'power_t'.
      effective_learning_rate = learning_rate_init / pow(t, power_t)
    - 'adaptive' keeps the learning rate constant to
      'learning_rate_init' as long as training loss keeps decreasing.
      Each time two consecutive epochs fail to decrease training loss by at
      least tol, or fail to increase validation score by at least tol if
      'early_stopping' is on, the current learning rate is divided by 5.
    Only used when ``solver='sgd'``.
learning_rate_init : float, default=0.001
    The initial learning rate used. It controls the step-size
    in updating the weights. Only used when solver='sgd' or 'adam'.
power_t : float, default=0.5
    The exponent for inverse scaling learning rate.
    It is used in updating effective learning rate when the learning_rate
    is set to 'invscaling'. Only used when solver='sgd'.
max_iter : int, default=200
    Maximum number of iterations. The solver iterates until convergence
    (determined by 'tol') or this number of iterations. For stochastic
    solvers ('sgd', 'adam'), note that this determines the number of epochs
    (how many times each data point will be used), not the number of
    gradient steps.
shuffle : bool, default=True
    Whether to shuffle samples in each iteration. Only used when
    solver='sgd' or 'adam'.
random_state : int, RandomState instance, default=None
    Determines random number generation for weights and bias
    initialization, train-test split if early stopping is used, and batch
    sampling when solver='sgd' or 'adam'.
    Pass an int for reproducible results across multiple function calls.
tol : float, default=1e-4
    Tolerance for the optimization. When the loss or score is not improving
    by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
    unless ``learning_rate`` is set to 'adaptive', convergence is
    considered to be reached and training stops.
verbose : bool, default=False
    Whether to print progress messages to stdout.
warm_start : bool, default=False
    When set to True, reuse the solution of the previous
    call to fit as initialization, otherwise, just erase the
    previous solution.
momentum : float, default=0.9
    Momentum for gradient descent update. Should be between 0 and 1. Only
    used when solver='sgd'.
nesterovs_momentum : bool, default=True
    Whether to use Nesterov's momentum. Only used when solver='sgd' and
    momentum > 0.
early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation
    score is not improving. If set to true, it will automatically set
    aside 10% of training data as validation and terminate training when
    validation score is not improving by at least tol for
    ``n_iter_no_change`` consecutive epochs. The split is stratified,
    except in a multilabel setting.
    If early stopping is False, then the training stops when the training
    loss does not improve by more than tol for n_iter_no_change consecutive
    passes over the training set.
    Only effective when solver='sgd' or 'adam'.
validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True.
beta_1 : float, default=0.9
    Exponential decay rate for estimates of first moment vector in adam,
    should be in [0, 1). Only used when solver='adam'.
beta_2 : float, default=0.999
    Exponential decay rate for estimates of second moment vector in adam,
    should be in [0, 1). Only used when solver='adam'.
epsilon : float, default=1e-8
    Value for numerical stability in adam. Only used when solver='adam'.
n_iter_no_change : int, default=10
    Maximum number of epochs to not meet ``tol`` improvement.
    Only effective when solver='sgd' or 'adam'.
max_fun : int, default=15000
    Only used when solver='lbfgs'. Maximum number of loss function calls.
    The solver iterates until convergence (determined by 'tol'), number
    of iterations reaches max_iter, or this number of loss function calls.
    Note that number of loss function calls will be greater than or equal
    to the number of iterations for the `MLPClassifier`.
"""

ActivationT = Literal["identity", "logistic", "tanh", "relu"]
SolverT = Literal["lbfgs", "sgd", "adam"]
LearningRateT = Literal["constant", "invscaling", "adaptive"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class MLPClassifierConfig(ModelConfig):
    hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [
            100,
        ],
    )
    activation: ActivationT = "relu"
    solver: SolverT = "adam"
    alpha: float = 0.0001
    batch_size: Union[int, Literal["auto"]] = "auto"
    learning_rate: LearningRateT = "constant"
    learning_rate_init: float = 0.001
    power_t: float = 0.5
    max_iter: int = 200
    shuffle: bool = True
    random_state: Union[None, int, RandomState] = None
    tol: float = 1e-4
    verbose: bool = False
    warm_start: bool = False
    momentum: float = 0.9
    nesterovs_momentum: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    n_iter_no_change: int = 10
    max_fun: int = 15000


class MLPClassifier(SKLearnClassificationMixin, SKLearnModelBase[MLPClassifierConfig]):
    @property
    def _config_builder(self) -> Type[MLPClassifierConfig]:
        return MLPClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKMLPClassifier:
        config = self.model_config
        return SKMLPClassifier(
            hidden_layer_sizes=tuple(config.hidden_layer_sizes),
            activation=config.activation,
            solver=config.solver,
            alpha=config.alpha,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            learning_rate_init=config.learning_rate_init,
            power_t=config.power_t,
            max_iter=config.max_iter,
            shuffle=config.shuffle,
            random_state=config.random_state,
            tol=config.tol,
            verbose=config.verbose,
            warm_start=config.warm_start,
            momentum=config.momentum,
            nesterovs_momentum=config.nesterovs_momentum,
            early_stopping=config.early_stopping,
            validation_fraction=config.validation_fraction,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
            epsilon=config.epsilon,
            n_iter_no_change=config.n_iter_no_change,
            max_fun=config.max_fun,
        )
