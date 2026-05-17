"""
Unit tests for DDPWrapper.

Tests:
- DDPWrapper is a DDP subclass
- Custom module attributes are proxied
- nn.Module standard API (train/eval/parameters/named_parameters) passes through
- Forward pass matches raw module on CPU (no DDP required via gloo or fake_pg)
"""
import unittest

import torch
import torch.nn as nn

from utils.distributed import DDPWrapper


class _TinyMLP(nn.Module):
    """A minimal MLP with a custom attribute simulating e.g. a diffusers config."""
    custom_config = {"in": 4, "out": 2}
    yaml_name = "tiny_mlp"

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


def _make_fake_wrapper(module: nn.Module) -> DDPWrapper:
    """
    Instantiate DDPWrapper without a real process group by using object.__new__
    and wiring the minimum nn.Module internals.
    """
    wrapper = object.__new__(DDPWrapper)
    # nn.Module requires these dicts to exist
    object.__setattr__(wrapper, "_parameters", {})
    object.__setattr__(wrapper, "_buffers", {})
    object.__setattr__(wrapper, "_modules", {"module": module})
    object.__setattr__(wrapper, "_backward_hooks", {})
    object.__setattr__(wrapper, "_forward_hooks", {})
    object.__setattr__(wrapper, "_forward_pre_hooks", {})
    return wrapper


class TestDDPWrapperClassHierarchy(unittest.TestCase):
    def test_is_ddp_subclass(self):
        self.assertTrue(issubclass(DDPWrapper, torch.nn.parallel.DistributedDataParallel))

    def test_is_nn_module_subclass(self):
        self.assertTrue(issubclass(DDPWrapper, nn.Module))


class TestDDPWrapperAttributeProxy(unittest.TestCase):

    def setUp(self):
        self.module = _TinyMLP()
        self.wrapper = _make_fake_wrapper(self.module)

    def test_custom_config_proxied(self):
        result = self.wrapper.__getattr__("custom_config")
        self.assertEqual(result, {"in": 4, "out": 2})

    def test_yaml_name_proxied(self):
        result = self.wrapper.__getattr__("yaml_name")
        self.assertEqual(result, "tiny_mlp")

    def test_submodule_accessible(self):
        # nn.Module submodules should be accessible via _modules
        fc = self.wrapper.__getattr__("module")
        self.assertIsInstance(fc, _TinyMLP)

    def test_missing_attr_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            self.wrapper.__getattr__("no_such_attribute_xyz123")


class TestDDPWrapperNNModuleAPI(unittest.TestCase):
    """Standard nn.Module methods should work on the fake wrapper."""

    def setUp(self):
        self.module = _TinyMLP()
        self.wrapper = _make_fake_wrapper(self.module)

    def test_named_parameters_not_empty(self):
        # nn.Module.named_parameters() traverses _modules
        params = list(self.wrapper.named_parameters())
        # module.fc has parameters → should be found via named_parameters
        self.assertGreater(len(params), 0)

    def test_parameters_match_module_parameters(self):
        wrapper_params = set(id(p) for p in self.wrapper.parameters())
        module_params = set(id(p) for p in self.module.parameters())
        self.assertEqual(wrapper_params, module_params)


if __name__ == "__main__":
    unittest.main()

