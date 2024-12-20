from collections import OrderedDict

from lightGE.core.tensor import Tensor, conv2d, max_pool2d, avg_pool2d
import numpy as np
import lightGE.core.init as init

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Model(object):
    def __init__(self, device="cpu"):
        self.is_eval = False
        self.sub_models = OrderedDict()
        self.params = OrderedDict()
        self.training = True
        self.device = device

    def add(self, layer):
        self.sub_models[str(len(self.sub_models))] = layer
     
    @staticmethod
    def _is_device(device):
        if device == 'cpu' or device == 'gpu':
            return True
        return False

    def to(self, device):
        if not self._is_device(device):
            raise Exception(device + "is not a legal device")
        self.device = device
        for key, value in self.__dict__.items():
            if isinstance(value, Model):
                value.to(device)
            if isinstance(value, Tensor):
                value.to(device)
        if len(self.sub_models) > 0:
            for layer in self.sub_models.values():
                layer.to(device)

    def parameters(self):
        return self.__parameters(set())

    def __parameters(self, visited):
        visited.add(id(self))
        params = list(self.params.values())
        for l in self.sub_models.values():
            if id(l) not in visited:
                params += l.__parameters(visited)

        return params

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def eval(self):
        self.training = False
        self.__eval(set())

    def __eval(self, visited):
        visited.add(id(self))
        self.is_eval = True
        for l in self.sub_models.values():
            if id(l) not in visited:
                l.__eval(visited)

    def train(self):
        self.training = True
        self.__train(set())

    def __train(self, visited):
        visited.add(id(self))
        self.is_eval = False
        for l in self.sub_models.values():
            if id(l) not in visited:
                l.__train(visited)

    def __setattr__(self, key, value):
        if isinstance(value, Model):
            self.sub_models[key] = value

        if isinstance(value, Tensor) and value.autograd:
            self.params[key] = value

        super().__setattr__(key, value)

    def _extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self._extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []

        for key, value in self.params.items():
            mod_str = _addindent(str(value.shape), 2)
            child_lines.append('[' + key + ']: ' + mod_str)

        for key, module in self.sub_models.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


class Sequential(Model):
    def __init__(self, layers):
        super().__init__()
        for l in layers:
            self.add(l)
        
    def add(self, layer):
        self.sub_models[str(len(self.sub_models))] = layer

    def forward(self, input: Tensor):
        for layer in self.sub_models.values():
            input = layer.forward(input)
        return input


class ModuleList(Model):
    def __init__(self, layers):
        super().__init__()
        self.list = layers
        for l in layers:
            self.add(l)
              
    def append(self, layer):
        self.list.append(layer)
        self.add(layer)

    def __iter__(self):
        return iter(self.list)  
    
    def forward(self, input: Tensor):
        for layer in self.list:
            input = layer(input) + input
        return input


class Linear(Model):
    def __init__(self, n_inputs, n_outputs, bias = True):
        super().__init__()
        self.n_outputs = n_outputs
        self.bias = bias
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(W, autograd=True)
        
        if bias:
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)

    def forward(self, input: Tensor):
        shape = input.shape
        input = input.reshape((-1,shape[-1]))

        my_list = list(shape)
        my_list[-1] = self.n_outputs
        out_shape = tuple(my_list)
        
        if self.bias:
            return (input.mm(self.weight) + self.bias).reshape(out_shape)
        return input.mm(self.weight).reshape(out_shape)


class Conv2d(Model):
    def __init__(self, n_inputs, n_outputs, filter_size, stride=1, padding=0, bias=False, init_method = 'kaiming'):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.init_method = init_method
        self.weight = Tensor(self.initialize_weights((n_outputs, n_inputs, filter_size, filter_size), self.init_method), autograd=True)

        if bias:
            self.bias = Tensor(init.zero_init(n_outputs), autograd=True)
        else:
            self.bias = None

    # 初始化权重的函数
    def initialize_weights(self, shape, method):
        if method == 'kaiming':
            return init.kaiming_init(shape)
        if method == 'lecun':
            return init.lecun_init(shape)
        if method == 'normal':
            return init.normal_init(shape)
        if method == 'uniform':
            return init.uniform_init(shape)
        else:
            raise Exception(method + " is not a legal initialization method")

    def forward(self, input: Tensor):
        output = conv2d(input, self.weight, stride=self.stride, padding=self.padding)

        if self.bias is not None:
            output += self.bias.reshape((1, self.bias.shape[0], 1, 1))

        return output


class MaxPool2d(Model):
    def __init__(self, filter_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.filter_size = filter_size
        if stride is None:
            self.stride = filter_size
        else:
            self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        return max_pool2d(input, self.filter_size, self.stride, self.padding, self.ceil_mode)


class AvgPool2d(Model):
    def __init__(self, filter_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.filter_size = filter_size
        if stride is None:
            self.stride = filter_size
        else:
            self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        return avg_pool2d(input, self.filter_size, self.stride, self.padding, self.ceil_mode)

class AdaptiveAvgPool2d(Model):
    def __init__(self, filter_size = 1, stride = 1, padding=0, ceil_mode=False):
        super().__init__()
        self.filter_size = filter_size
        if stride is None:
            self.stride = filter_size
        else:
            self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        return avg_pool2d(input, input.shape[2], self.stride, self.padding, self.ceil_mode)
    
class LSTM(Model):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.W_ih = Tensor(np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs), autograd=True)
        self.W_hh = Tensor(np.random.randn(n_hidden, n_hidden) * np.sqrt(2.0 / n_hidden), autograd=True)
        self.W_hy = Tensor(np.random.randn(n_hidden, n_outputs) * np.sqrt(2.0 / n_hidden), autograd=True)

        self.bias_h = Tensor(np.zeros(n_hidden), autograd=True)
        self.bias_y = Tensor(np.zeros(n_outputs), autograd=True)

    def forward(self, input: Tensor, hidden: Tensor):
        self.hidden = hidden
        self.input = input

        self.ih = input.mm(self.W_ih) + self.bias_h
        self.hh = self.hidden.mm(self.W_hh)
        self.h = self.ih + self.hh
        self.h.tanh()

        self.y = self.h.mm(self.W_hy) + self.bias_y
        self.y.softmax(dim=1)

        return self.y, self.h


class Tanh(Model):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor):
        return input.tanh()


class Sigmoid(Model):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor):
        return input.sigmoid()


class ReLU(Model):
    def __int__(self):
        super().__init__()

    def forward(self, input: Tensor):
        return input.relu()
    
class LeakyReLU(Model):
    def __init__(self, alpha = 0.01):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, input: Tensor):
        return input.leakyrelu(self.alpha)


class BatchNorm1d(Model):
    def __init__(self, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs

        self.gamma = Tensor(np.ones(n_inputs), autograd=True)
        self.beta = Tensor(np.zeros(n_inputs), autograd=True)
        self.eps = Tensor(1e-8, autograd=False)

    def forward(self, input: Tensor):
  
        mean_val = input.mean(0)
        var_val = input.var(0).add(self.eps)
        std_val = var_val.sqrt()

        norm = (input - mean_val) / std_val

        return self.gamma * norm + self.beta


class BatchNorm2d(Model):
    def __init__(self, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs

        self.gamma = Tensor(np.ones(n_inputs), autograd=True)
        self.beta = Tensor(np.zeros(n_inputs), autograd=True)
        self.eps = Tensor(1e-8, autograd=False)

    def forward(self, input: Tensor):
        mean_val = input.mean(0).unsqueeze(0)
        var_val = input.var(0) + self.eps
        std_val = var_val.sqrt().unsqueeze(0)
        norm = (input - mean_val) / std_val
        return self.gamma.reshape((1, -1, 1, 1))  * norm + self.beta.reshape((1, -1, 1, 1)) 

class Dropout(Model):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor):
        if self.is_eval:
            return input
        else:
            self.mask = Tensor(np.random.binomial(1, 1 - self.p, size=input.data.shape), autograd=True).to(self.device)
            return input * self.mask


class Dropout2d(Model):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor):
        if self.is_eval:
            return input
        else:
            self.mask = Tensor(np.random.binomial(1, 1 - self.p, size=(*input.data.shape[:-2], 1, 1)),
                               autograd=True).to(self.device)
            return input * self.mask


class LayerNorm(Model):
    def __init__(self, n_inputs):
        super().__init__()     
        self.n_inputs = n_inputs
        self.gamma = Tensor(np.ones(n_inputs), autograd=True)
        self.beta = Tensor(np.zeros(n_inputs), autograd=True)
        self.eps = Tensor(1e-8, autograd=False)
        
    def forward(self, input: Tensor):     
        mean_val = input.mean(1).unsqueeze(1)
        var_val = input.var(1)+self.eps
        std_val = var_val.sqrt().unsqueeze(1)
        norm = (input - mean_val) / std_val
        return self.gamma * norm + self.beta
