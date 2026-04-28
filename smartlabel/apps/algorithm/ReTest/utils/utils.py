import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from json import dump
import random
from PIL import Image
from numpy.testing import assert_array_almost_equal
import scipy # type: ignore
from typing import Optional, Union, Any, Tuple

from utils.logger import setup_color_logger
logger = setup_color_logger()


def init_seeds(seed: int=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # type: ignore
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.cuda.empty_cache()


def set_device(gpu: Optional[Union[int, str]]=None):
    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        logger.info(f'Available GPUs Index : {os.environ["CUDA_VISIBLE_DEVICES"]} ,total : {torch.cuda.device_count()}')
    except KeyError:
        logger.info('No GPU available, using CPU ... ')
    return torch.device('cuda') if torch.cuda.device_count() >= 1 else torch.device('cpu')


def save_checkpoint(state: dict[str, Any], filename: str='checkpoint.pth'):
    torch.save(state, filename)


def init_weights(module: nn.Module, init_method: str='He'):
    for _, m in module.named_modules(): # pyright: ignore[reportUnknownVariableType]
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)


def save_params(params: Any, params_file: Any, json_format: bool=False):
    with open(params_file, 'w') as f:
        if not json_format:
            params_file.replace('.json', '.txt')
            for k, v in params.__dict__.items():
                f.write(f'{k:<20}: {v}\n')
        else:
            params_file.replace('.txt', '.json')
            dump(params.__dict__, f, indent=4)


def save_config(params: Any, params_file: str):
    config_file_path = params.cfg_file
    shutil.copy(config_file_path, params_file)


def save_network_info(model: nn.Module, path: str):
    with open(path, 'w') as f:
        f.writelines(model.__repr__())


def str_is_int(x: str) -> bool:
    if x.count('-') > 1:
        return False
    if x.isnumeric():
        return True
    if x.startswith('-') and x.replace('-', '').isnumeric():
        return True
    return False


def str_is_float(x: str) -> bool:
    if str_is_int(x):
        return False
    try:
        _ = float(x)
        return True
    except ValueError:
        return False


class Config(object):
    def set_item(self, key: str, value: Any):
        if isinstance(value, str):
            if str_is_int(value):
                value = int(value)
            elif str_is_float(value):
                value = float(value)
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
        if key.endswith('milestones'):
            try:
                tmp_v = value[1:-1].split(',')
                value = list(map(int, tmp_v))
            except:
                raise AssertionError(f'{key} is: {value}, format not supported!')
        self.__dict__[key] = value

    def __repr__(self):
        # return self.__dict__.__repr__()
        ret = 'Config:\n{\n'
        for k in self.__dict__.keys():
            s = f'    {k}: {self.__dict__[k]}\n'
            ret += s
        ret += '}\n'
        return ret


def load_from_cfg(path: str) -> Config:
    cfg = Config()
    if not path.endswith('.cfg'):
        path = path + '.cfg'
    if not os.path.exists(path) and os.path.exists('config' + os.sep + path):
        path = 'config' + os.sep + path
    assert os.path.isfile(path), f'{path} is not a valid config file.'

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.strip() for x in lines]

    for line in lines:
        if line.startswith('['):
            continue
        k, v = line.replace(' ', '').split('=')
        
        cfg.set_item(key=k, value=v)
    cfg.set_item(key='cfg_file', value=path)

    return cfg

def variable_to_numpy(x: Union[torch.Tensor, np.ndarray[Any, Any]]) -> Union[np.ndarray[Any, Any], float]:
    if type(x) == np.ndarray:
        return x
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        return float(np.sum(ans))
    return ans

def list_dir(root: str, prefix: bool=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories

def list_files(root: str, suffix: Union[str, Tuple[str, ...]], prefix: bool=False) -> list[str]:
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


# basic function
def multiclass_noisify(y: np.ndarray, P: np.ndarray, random_state: int=0) -> np.ndarray:
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def generate_noise_matrix(noise_type: str, closeset_noise_ratio: float, openset_noise_ratio: float=0.8, nb_classes: int=10) -> np.ndarray:
    """

    Example of the noise transition matrix (closeset_ratio = 0.3):
        - Symmetric:
            -                               -
            | 0.7  0.1  0.1  0.1  0.0  0.0  |
            | 0.1  0.7  0.1  0.1  0.0  0.0  |
            | 0.1  0.1  0.7  0.1  0.0  0.0  |
            | 0.1  0.1  0.1  0.7  0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            -                               -
        - Pairflip
            -                               -
            | 0.7  0.3  0.0  0.0  0.0  0.0  |
            | 0.0  0.7  0.3  0.0  0.0  0.0  |
            | 0.0  0.0  0.7  0.3  0.0  0.0  |
            | 0.3  0.0  0.0  0.7  0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            -                               -

    """
    # assert closeset_noise_ratio > 0.0, 'noise rate must be greater than 0.0'
    assert 0.0 <= openset_noise_ratio < 1.0, 'the ratio of out-of-distribution class must be within [0.0, 1.0)'
    closeset_nb_classes = int(nb_classes * (1 - openset_noise_ratio))
    # openset_nb_classes = nb_classes - closeset_nb_classes
    if noise_type == 'symmetric':
        P = np.ones((nb_classes, nb_classes))
        P = (closeset_noise_ratio / (closeset_nb_classes - 1)) * P
        for i in range(closeset_nb_classes):
            P[i, i] = 1.0 - closeset_noise_ratio
        for i in range(closeset_nb_classes, nb_classes):
            P[i, :] = 1.0 / closeset_nb_classes
        for i in range(closeset_nb_classes, nb_classes):
            P[:, i] = 0.0
    elif noise_type == 'pairflip':
        P = np.eye(nb_classes)
        P[0, 0], P[0, 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        for i in range(1, closeset_nb_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        P[closeset_nb_classes - 1, closeset_nb_classes - 1] = 1.0 - closeset_noise_ratio
        P[closeset_nb_classes - 1, 0] = closeset_noise_ratio
        for i in range(closeset_nb_classes, nb_classes):
            P[i, :] = 1.0 / closeset_nb_classes
        for i in range(closeset_nb_classes, nb_classes):
            P[:, i] = 0.0
    else:
        raise AssertionError("noise type must be either symmetric or pairflip")
    return P


def noisify(y_train, noise_transition_matrix, random_state=None):
    y_train_noisy = multiclass_noisify(y_train, P=noise_transition_matrix, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    # assert actual_noise > 0.0
    return y_train_noisy, actual_noise

def noisify_dataset(nb_classes = 10, train_data = None, train_labels = None, noise_type=None,
                    closeset_noise_ratio=0.0, openset_noise_ratio=0.0, random_state=0, ret_matrix = False):
    logger.debug(f'noisify with {noise_type} label noise',style='bold', color='blue')
    if noise_type in ['symmetric', 'pairflip']:
        noise_transition_matrix = generate_noise_matrix(noise_type, closeset_noise_ratio, openset_noise_ratio, nb_classes)
        train_noisy_labels, actural_noise_rate = noisify(train_labels, noise_transition_matrix, random_state)
        if not ret_matrix:
            return train_noisy_labels
        else: 
            if openset_noise_ratio > 0.0:
                extended_T = np.zeros((noise_transition_matrix.shape[0] + 1,noise_transition_matrix.shape[1]))
                extended_T[:nb_classes, :]=noise_transition_matrix
                extended_T[nb_classes,: ]=1 / nb_classes
                return train_noisy_labels, extended_T
            else:
                return train_noisy_labels, noise_transition_matrix
    elif noise_type == 'instance':
        feature_size = torch.Tensor(train_data[0]).reshape(-1).size(0)
        train_noisy_labels, actural_noise_rate = noisify_instance(train_data = train_data, train_labels = train_labels,\
            noise_rate = closeset_noise_ratio, feature_size = feature_size)
        noise_transition_matrix = None
        if not ret_matrix:
            return train_noisy_labels
        else: 
            return train_noisy_labels, None
    else:
        raise NotImplementedError(f'not support noise type \"{noise_type}\"')
    
def noisify_instance(train_data, train_labels, noise_rate, feature_size):
    if max(train_labels)>10:
        num_class = 100
    else:
        num_class = 10
    np.random.seed(0)

    q_ = np.random.normal(loc=noise_rate,scale=0.1,size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q)==80000:
            break

    w = np.random.normal(loc=0, scale=1, size=(feature_size, num_class))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = sample.flatten()
        p_all = np.matmul(sample,w)
        p_all[train_labels[i]] = -1000000
        p_all = q[i]* F.softmax(torch.tensor(p_all),dim=0).numpy()

        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class), p=p_all/sum(p_all)))

    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum()) / train_data.shape[0]
    ind = torch.tensor(train_labels).eq(torch.tensor(noisy_labels))

    return np.array(noisy_labels), over_all_noise_rate

def count_parameters_in_MB(model : nn.Module) -> float:
    ans = sum(np.prod(v.size()) for _, v in model.named_parameters())/1e6
    logger.info('total parameters in the network are {} M'.format(ans))
    return ans

eps = 1e-12

def Distance_squared(x, y, featdim=1):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d[torch.eye(d.shape[0]) == 1] = eps
    return d

def CalPairwise(dist):
    dist[dist < 0] = 0
    Pij = torch.exp(-dist)
    return Pij

# using student_t distribution to replace gaussian distribution
def CalPairwise_t(dist, v):
    C = scipy.special.gamma((v + 1) / 2) / (np.sqrt(v * np.pi) * scipy.special.gamma(v / 2))
    return torch.pow((1 + torch.pow(dist, 2) / v), - (v + 1) / 2)

        
def get_numpy(file_name):
    ret = None
    with open(file_name, 'rb') as f:
        ret = np.load(f, allow_pickle=True)
    return ret

def save_numpy(filename, array):
    if not (os.path.exists(os.path.dirname(filename))):
        os.mkdir(os.path.dirname(filename))
    np.save(filename, array)
    
# https://github.com/kuangliu/pytorch-retinanet/blob/master/transform.py
def resize(img, size, max_size=1000):
    '''Resize the input PIL image to the given size.
    Args:
      img: (PIL.Image) image to be resized.
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w, h)
        sw = sh = float(size) / size_min

        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow, oh), Image.BICUBIC)
