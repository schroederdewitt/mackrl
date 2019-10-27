
# Check if DEBUG mode is enabled (e.g. this is the debugger used by PyCharm)
try:
    import pydevd
    IS_PYDEV_DEBUG = True
    IS_PYCHARM_DEBUG = True
except ImportError:
    IS_PYDEV_DEBUG = False
    IS_PYCHARM_DEBUG = False

def to_np(tensor):
    """
    Converts a tensor of format bs*t*v into a list of 2D numpy arrays
    This is meant for DEBUGGING (numpy 2D lists can be easily viewed in PyCharm)
    """
    assert len(tensor.shape) == 3, "tensor needs to be of dimension 3!"
    tensor_np = tensor.cpu().numpy()
    np_list = []
    for _bs in range(tensor.shape[0]):
        np_list.append(tensor_np[_bs, :, :])
    return np_list

def to_pd(sequence_buffer):
    return sequence_buffer.to_pd()