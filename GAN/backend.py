import torch


def get_default_device(force_to_cpu=False):
    if force_to_cpu:
        print("Set default device to cpu.")
        return torch.device('cpu')

    if torch.cuda.is_available():
        print("Set default device to cuda.")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Set default device to mps.")
        return torch.device('mps')
    elif torch.xpu.is_available():
        print("Set default device to xpu.")
        return torch.device('xpu')
    else:
        print("Set default device to cpu.")
        return torch.device('cpu')


def empty_cache(device):
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()
    elif device == 'xpu':
        torch.xpu.empty_cache()
    else:
        pass


