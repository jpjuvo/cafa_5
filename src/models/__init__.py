import importlib

__all__ = [
    'get_model'
]

def get_model(model_fn, input_shape, num_of_labels, **kwargs):
    model_module = importlib.import_module(f"models.{model_fn}", package=None)
    return model_module.Model(input_shape=input_shape, num_of_labels=num_of_labels, **kwargs)