from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

from pydantic.dataclasses import dataclass

import datasets

if TYPE_CHECKING:
    import numpy as np

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelBase, ModelConfig
from molflux.modelzoo.models.sklearn import SKLearnClassificationMixin
from molflux.modelzoo.typing import PredictionResult
from molflux.modelzoo.utils import (
    get_concatenated_array,
    pick_features,
    validate_features,
)

try:
    from catboost import CatBoostClassifier as CBC
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("catboost", e) from e


_DESCRIPTION = """
Implementation of the scikit-learn API for a CatBoostClassifier.

CatBoost is based on gradient boosted decision trees. During training, a set
of decision trees is built consecutively. Each successive tree is built with
reduced loss compared to the previous trees.

The number of trees is controlled by the starting parameters.
To prevent overfitting, use the overfitting detector. When it is triggered,
trees stop being built.

Processing of categorical data is treated differently from other gradient tree based
methods, thus this type of model is recommended when dealing with categorical features.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
iterations : Optional[int], default: None
learning_rate: Optional[float], default: None
depth: Optional[int], default: None
l2_leaf_reg: Optional[int], default: None
model_size_reg: Optional[float], default: None
rsm: Optional[int], default: None
loss_function: Optional[Literal["Logloss", "CrossEntropy"]], default: None
border_count: Optional[int], default: None
feature_border_type: Optional[str], default: None
per_float_feature_quantization: Optional[List[str]], default: None
input_borders: Optional[str], default: None
output_borders: Optional[str], default: None
fold_permutation_block: Optional[int], default: None
od_pval: Optional[float], default: None
od_wait: Optional[int], default: None
od_type: Optional[str], default: None
nan_mode: Optional[str], default: None
counter_calc_method: Optional[str], default: None
leaf_estimation_iterations: Optional[int], default: None
leaf_estimation_method: Optional[str], default: None
thread_count: Optional[int], default: None
random_seed: Optional[int], default: None
use_best_model: Optional[bool], default: None
verbose: Optional[bool], default: None
logging_level: Optional[Union[int, str]], default: "Silent"
metric_period: Optional[int], default: None
ctr_leaf_count_limit: Optional[int], default: None
store_all_simple_ctr: Optional[bool], default: None
max_ctr_complexity: Optional[int], default: None
has_time: Optional[bool], default: None
allow_const_label: Optional[bool], default: None
classes_count: Optional[Any], default: None
class_weights: Optional[Any], default: None
one_hot_max_size: Optional[int], default: None
random_strength: Optional[int], default: None
catboost_name: Optional[str], default: None
ignored_features: Optional[Union[List[int], List[str]]], default: None
train_dir: Optional[str], default: None
custom_loss: Optional[str], default: None
custom_metric: Optional[str], default: None
eval_metric: Optional[str], default: None
bagging_temperature: Optional[float], default: None
save_snapshot: Optional[bool], default: None
snapshot_file: Optional[str], default: None
snapshot_interval: Optional[int], default: None
fold_len_multiplier: Optional[float], default: None
used_ram_limit: Optional[float], default: None
gpu_ram_part: Optional[float], default: None
allow_writing_files: Optional[bool], default: None
final_ctr_computation_mode: Optional[str], default: None
approx_on_full_history: Optional[bool], default: None
boosting_type: Optional[str], default: None
simple_ctr: Optional[bool], default: None
combinations_ctr: Optional[str], default: None
per_feature_ctr: Optional[str], default: None
task_type: Optional[str], default: None
device_config: Optional[str], default: None
devices: Optional[str], default: None
bootstrap_type: Optional[str], default: None
subsample: Optional[bool], default: None
sampling_unit: Optional[str], default: None
dev_score_calc_obj_block_size: Optional[int], default: None
max_depth: Optional[int], default: None
n_estimators: Optional[int], default: None
num_boost_round: Optional[int], default: None
num_trees: Optional[int], default: None
colsample_bylevel: Optional[int], default: None
random_state: Optional[int], default: None
reg_lambda: Optional[float], default: None
objective: Optional[str], default: None
eta: Optional[float], default: None
max_bin: Optional[int], default: None
gpu_cat_features_storage: Optional[str], default: None
data_partition: Optional[str], default: None
metadata: Optional[Dict], default: None
early_stopping_rounds: Optional[int], default: None
cat_features: Optional[List[str]], default: None
grow_policy: Optional[str], default: None
min_data_in_leaf: Optional[int], default: None
min_child_samples: Optional[int], default: None
max_leaves: Optional[int], default: None
num_leaves: Optional[int], default: None
score_function: Optional[str], default: None
leaf_estimation_backtracking: Optional[str], default: None
ctr_history_unit: Optional[int], default: None
monotone_constraints: Optional[Union[List[str], str, Dict, List]], default: None
feature_weights: Optional[Union[List, str, Dict]], default: None
penalties_coefficient: Optional[float], default: None
first_feature_use_penalties: Optional[bool], default: None
model_shrink_rate: Optional[float], default: None
model_shrink_mode: Optional[str], default: None
langevin: Optional[bool], default: None
diffusion_temperature: Optional[float], default: None
posterior_sampling: Optional[bool], default: None
boost_from_average: Optional[bool], default: None
dictionaries: Optional[Any], default: None
feature_calcers: Optional[Any], default: None
scale_pos_weight: Optional[Any], default: None
text_features: Optional[Any], default: None
text_processing: Optional[Any], default: None
tokenizers: Optional[Any], default: None
"""

LossFunctions = Literal["Logloss", "CrossEntropy"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class CatBoostClassifierConfig(ModelConfig):
    iterations: Optional[int] = None
    learning_rate: Optional[float] = None
    depth: Optional[int] = None
    l2_leaf_reg: Optional[int] = None
    model_size_reg: Optional[float] = None
    rsm: Optional[int] = None
    loss_function: Optional[LossFunctions] = None
    border_count: Optional[int] = None
    feature_border_type: Optional[str] = None
    per_float_feature_quantization: Optional[List[str]] = None
    input_borders: Optional[str] = None
    output_borders: Optional[str] = None
    fold_permutation_block: Optional[int] = None
    od_pval: Optional[float] = None
    od_wait: Optional[int] = None
    od_type: Optional[str] = None
    nan_mode: Optional[str] = None
    counter_calc_method: Optional[str] = None
    leaf_estimation_iterations: Optional[int] = None
    leaf_estimation_method: Optional[str] = None
    thread_count: Optional[int] = None
    random_seed: Optional[int] = None
    use_best_model: Optional[bool] = None
    verbose: Optional[bool] = None
    logging_level: Optional[Union[int, str]] = "Silent"
    metric_period: Optional[int] = None
    ctr_leaf_count_limit: Optional[int] = None
    store_all_simple_ctr: Optional[bool] = None
    max_ctr_complexity: Optional[int] = None
    has_time: Optional[bool] = None
    allow_const_label: Optional[bool] = None
    one_hot_max_size: Optional[int] = None
    random_strength: Optional[int] = None
    catboost_name: Optional[str] = None
    ignored_features: Optional[Union[List[int], List[str]]] = None
    train_dir: Optional[str] = None
    custom_metric: Optional[str] = None
    eval_metric: Optional[str] = None
    bagging_temperature: Optional[float] = None
    save_snapshot: Optional[bool] = None
    snapshot_file: Optional[str] = None
    snapshot_interval: Optional[int] = None
    fold_len_multiplier: Optional[float] = None
    used_ram_limit: Optional[float] = None
    gpu_ram_part: Optional[float] = None
    allow_writing_files: Optional[bool] = None
    final_ctr_computation_mode: Optional[str] = None
    approx_on_full_history: Optional[bool] = None
    boosting_type: Optional[str] = None
    simple_ctr: Optional[bool] = None
    combinations_ctr: Optional[str] = None
    per_feature_ctr: Optional[str] = None
    task_type: Optional[str] = None
    device_config: Optional[str] = None
    devices: Optional[str] = None
    bootstrap_type: Optional[str] = None
    subsample: Optional[bool] = None
    sampling_unit: Optional[str] = None
    dev_score_calc_obj_block_size: Optional[int] = None
    max_depth: Optional[int] = None
    n_estimators: Optional[int] = None
    num_boost_round: Optional[int] = None
    num_trees: Optional[int] = None
    colsample_bylevel: Optional[int] = None
    random_state: Optional[int] = None
    reg_lambda: Optional[float] = None
    objective: Optional[str] = None
    eta: Optional[float] = None
    max_bin: Optional[int] = None
    gpu_cat_features_storage: Optional[str] = None
    data_partition: Optional[str] = None
    metadata: Optional[Dict] = None
    early_stopping_rounds: Optional[int] = None
    cat_features: Optional[List[str]] = None
    grow_policy: Optional[str] = None
    min_data_in_leaf: Optional[int] = None
    min_child_samples: Optional[int] = None
    max_leaves: Optional[int] = None
    num_leaves: Optional[int] = None
    score_function: Optional[str] = None
    leaf_estimation_backtracking: Optional[str] = None
    ctr_history_unit: Optional[int] = None
    monotone_constraints: Optional[Union[List[str], str, Dict, List]] = None
    feature_weights: Optional[Union[List, str, Dict]] = None
    penalties_coefficient: Optional[float] = None
    first_feature_use_penalties: Optional[bool] = None
    model_shrink_rate: Optional[float] = None
    model_shrink_mode: Optional[str] = None
    langevin: Optional[bool] = None
    diffusion_temperature: Optional[float] = None
    posterior_sampling: Optional[bool] = None
    boost_from_average: Optional[bool] = None
    auto_class_weights: Optional[str] = None
    dictionaries: Optional[Any] = None
    feature_calcers: Optional[Any] = None
    scale_pos_weight: Optional[Any] = None
    text_features: Optional[Any] = None
    text_processing: Optional[Any] = None
    tokenizers: Optional[Any] = None


class CatBoostClassifier(
    SKLearnClassificationMixin,
    ModelBase[CatBoostClassifierConfig],
):
    """
    Note:
        The cat_features must specify the column names of the input Dataset object. If
        the column is a column of arrays (for example a column of fingerprints), it
        is assumed that each array molflux is a categorical column of its own.
    """

    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[CatBoostClassifierConfig]:
        return CatBoostClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> CBC:
        """Note: Not yet pulling the config in"""
        config: CatBoostClassifierConfig = self.model_config
        return CBC(
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            model_size_reg=config.model_size_reg,
            rsm=config.rsm,
            loss_function=config.loss_function,
            border_count=config.border_count,
            feature_border_type=config.feature_border_type,
            per_float_feature_quantization=config.per_float_feature_quantization,
            input_borders=config.input_borders,
            output_borders=config.output_borders,
            fold_permutation_block=config.fold_permutation_block,
            od_pval=config.od_pval,
            od_wait=config.od_wait,
            od_type=config.od_type,
            nan_mode=config.nan_mode,
            counter_calc_method=config.counter_calc_method,
            leaf_estimation_iterations=config.leaf_estimation_iterations,
            leaf_estimation_method=config.leaf_estimation_method,
            thread_count=config.thread_count,
            random_seed=config.random_seed,
            use_best_model=config.use_best_model,
            verbose=config.verbose,
            logging_level=config.logging_level,
            metric_period=config.metric_period,
            ctr_leaf_count_limit=config.ctr_leaf_count_limit,
            store_all_simple_ctr=config.store_all_simple_ctr,
            max_ctr_complexity=config.max_ctr_complexity,
            has_time=config.has_time,
            allow_const_label=config.allow_const_label,
            one_hot_max_size=config.one_hot_max_size,
            random_strength=config.random_strength,
            name=config.catboost_name,
            ignored_features=config.ignored_features,
            train_dir=config.train_dir,
            custom_metric=config.custom_metric,
            eval_metric=config.eval_metric,
            bagging_temperature=config.bagging_temperature,
            save_snapshot=config.save_snapshot,
            snapshot_file=config.snapshot_file,
            snapshot_interval=config.snapshot_interval,
            fold_len_multiplier=config.fold_len_multiplier,
            used_ram_limit=config.used_ram_limit,
            gpu_ram_part=config.gpu_ram_part,
            allow_writing_files=config.allow_writing_files,
            final_ctr_computation_mode=config.final_ctr_computation_mode,
            approx_on_full_history=config.approx_on_full_history,
            boosting_type=config.boosting_type,
            simple_ctr=config.simple_ctr,
            combinations_ctr=config.combinations_ctr,
            per_feature_ctr=config.per_feature_ctr,
            task_type=config.task_type,
            device_config=config.device_config,
            devices=config.devices,
            bootstrap_type=config.bootstrap_type,
            subsample=config.subsample,
            sampling_unit=config.sampling_unit,
            dev_score_calc_obj_block_size=config.dev_score_calc_obj_block_size,
            max_depth=config.max_depth,
            n_estimators=config.n_estimators,
            num_boost_round=config.num_boost_round,
            num_trees=config.num_trees,
            colsample_bylevel=config.colsample_bylevel,
            random_state=config.random_state,
            reg_lambda=config.reg_lambda,
            objective=config.objective,
            eta=config.eta,
            max_bin=config.max_bin,
            gpu_cat_features_storage=config.gpu_cat_features_storage,
            data_partition=config.data_partition,
            metadata=config.metadata,
            early_stopping_rounds=config.early_stopping_rounds,
            cat_features=config.cat_features,
            grow_policy=config.grow_policy,
            min_data_in_leaf=config.min_data_in_leaf,
            min_child_samples=config.min_child_samples,
            max_leaves=config.max_leaves,
            num_leaves=config.num_leaves,
            score_function=config.score_function,
            leaf_estimation_backtracking=config.leaf_estimation_backtracking,
            ctr_history_unit=config.ctr_history_unit,
            monotone_constraints=config.monotone_constraints,
            feature_weights=config.feature_weights,
            penalties_coefficient=config.penalties_coefficient,
            first_feature_use_penalties=config.first_feature_use_penalties,
            model_shrink_rate=config.model_shrink_rate,
            model_shrink_mode=config.model_shrink_mode,
            langevin=config.langevin,
            diffusion_temperature=config.diffusion_temperature,
            posterior_sampling=config.posterior_sampling,
            boost_from_average=config.boost_from_average,
            auto_class_weights=config.auto_class_weights,
            dictionaries=config.dictionaries,
            feature_calcers=config.feature_calcers,
            scale_pos_weight=config.scale_pos_weight,
            text_features=config.text_features,
            text_processing=config.text_processing,
            tokenizers=config.tokenizers,
        )

    def _train(
        self,
        train_data: datasets.Dataset,
        **kwargs: Any,
    ) -> Any:
        """
        Note:
            This is not inheriting from SKLearn because of the special data treatment
        that needs to be done for categorical feature handling
        """

        # validate y features as well
        validate_features(train_data, self.y_features)

        x_data = pick_features(train_data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)

        y_data = pick_features(train_data, self.y_features)
        y = get_concatenated_array(y_data, self.y_features)

        # instantiate model
        self.model: CBC = self._instantiate_model()

        # train
        self.model.fit(X, y)

    def _predict(
        self,
        data: datasets.Dataset,
        **kwargs: Any,
    ) -> PredictionResult:
        """Note: same as _train, see above"""

        # TODO(avianello): for now catboost does not support multitask
        display_names = self._predict_display_names
        display_name = display_names[0]

        if not len(data):
            return {display_name: []}

        X = get_concatenated_array(data, self.x_features)
        y_predict: np.ndarray = self.model.predict(X)

        return {display_name: y_predict.tolist()}
