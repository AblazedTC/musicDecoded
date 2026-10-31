"""T-118: chord-model weights load once per process (fake submodule, no torch)."""
import types

from services.chord_cnn_lstm_service import cache_networks


class ChordNet:
    def __init__(self, _):
        pass


def fake_module():
    loads = []

    class NetworkInterface:
        def __init__(self, net, save_name, load_checkpoint=False, load_path="cache_data"):
            loads.append(save_name)  # stands in for torch.load of the checkpoint

    def chord_recognition():  # what the submodule does on every call: look up NetworkInterface by name
        return [module.NetworkInterface(ChordNet(None), name, load_checkpoint=False) for name in module.MODEL_NAMES]

    module = types.SimpleNamespace(NetworkInterface=NetworkInterface, ChordNet=ChordNet, MODEL_NAMES=["s0", "s1"],
                                   chord_recognition=chord_recognition)
    return module, loads


def test_weights_load_once_across_jobs():
    module, loads = fake_module()
    cache_networks(module)
    first = module.chord_recognition()
    second = module.chord_recognition()
    assert loads == ["s0", "s1"]
    assert all(a is b for a, b in zip(first, second))


def test_patching_twice_keeps_one_cache():
    module, loads = fake_module()
    cache_networks(module)
    cache_networks(module)
    module.chord_recognition()
    module.chord_recognition()
    assert loads == ["s0", "s1"]
