try:
    import torch
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning", e) from e

# NOTE Some schedulers disabled due to complex initialisation
# Would require a bit more custom logic (but could be done)


AVAILABLE_SCHEDULERS = {
    # "LambdaLR": torch.optim.lr_scheduler.LambdaLR,
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "ConstantLR": torch.optim.lr_scheduler.ConstantLR,
    "LinearLR": torch.optim.lr_scheduler.LinearLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    # "ChainedScheduler": torch.optim.lr_scheduler.ChainedScheduler,
    # "SequentialLR": torch.optim.lr_scheduler.SequentialLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
}
