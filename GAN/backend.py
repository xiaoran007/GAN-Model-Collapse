import torch


def get_default_device(force_to_cpu=False, force_skip_mps=False):
    if force_to_cpu:
        print("Set default device to cpu.")
        return torch.device('cpu')

    if check_cuda():
        print("Set default device to cuda.")
        return torch.device('cuda')
    elif check_mps() and not force_skip_mps:
        print("Set default device to mps.")
        return torch.device('mps')
    elif check_xpu():
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


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def check_cuda():
    try:
        if torch.cuda.is_available():
            return True
        else:
            return False
    except AttributeError:
        return False


def check_mps():
    try:
        if torch.backends.mps.is_available():
            return True
        else:
            return False
    except AttributeError:
        return False


def check_xpu():
    try:
        if torch.xpu.is_available():
            return True
        else:
            return False
    except AttributeError:
        return False
