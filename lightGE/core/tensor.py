import numpy as np
import scipy.special
from typing import Iterable, List
try:
    import cupy as cp
    if not cp.cuda.is_available():
        raise Exception()
except:
    print("You do not have cupy for gpu")
    cp = np
    
from einops import rearrange as einops_rearrange

class Tensor(object):
    def __init__(self, data,
                 autograd: bool = False,
                 creation_op=None,
                 device="cpu"):
        if device == "cpu":
            self.data = np.array(data, dtype=np.float64)
        else:
            self.data = cp.array(data, dtype=cp.float64)
            
        if autograd:
            if device == "cpu":
                self.grad = np.zeros_like(self.data)
            else:
                self.grad = cp.zeros_like(self.data)
        else:
            self.grad = None
            
        self.shape = self.data.shape
        self.autograd = autograd
        self.creation_op = creation_op
        self.dependents = {}

        self.tcg_id = TcGraph.AddTensor(self)
        self.device = device

    def all_children_grads_accounted_for(self):
        for cnt in self.dependents.values():
            if cnt != 0:
                return False
        return True
        
    
    @staticmethod
    def _is_device(device):
        if device == "cpu" or device == "gpu":
            return True
        return False

    def to(self, device):
        if device == self.device:
            return self
        if not self._is_device(device):
            raise Exception(device + "is not a legal device")
        self.device = device
        if device == "cpu":
            self.data = np.array(self.data.get(), dtype=np.float64)
        else:
            self.data = cp.array(self.data, dtype=cp.float64)
        if self.autograd:
            if device == "cpu":
                self.grad = np.array(self.grad.get(), dtype=np.float64)
            else:
                self.grad = cp.array(self.grad, dtype=cp.float64)
        return self

    def backward(self, grad=None, origin_id=None):
        if self.autograd:
            if grad is None:
                grad = np.ones_like(self.data)
            else:
                # print(self.creation_op.__class__)
                # print(grad.shape, grad.sum())
                self.grad += grad
                grad = self.grad
                if origin_id is not None:
                    if self.dependents[origin_id] == 0:
                        raise Exception("cannot backprop more than once")
                    self.dependents[origin_id] -= 1

            if self.all_children_grads_accounted_for():
                if self.creation_op is not None:
                    self.creation_op.backward(grad)

    def __add__(self, other):
        op: Op = AddOp(self, other, compute_mode=self.device)
        return op.calc()

    def __neg__(self):
        op: Op = NegOp(self, compute_mode=self.device)
        return op.calc()

    def __sub__(self, other):
        op: Op = SubOp(self, other, compute_mode=self.device)
        return op.calc()

    def __mul__(self, other):
        op: Op = MulOp(self, other, compute_mode=self.device)
        return op.calc()

    def __truediv__(self, other):
        op: Op = DivOp(self, other, compute_mode=self.device)
        return op.calc()

    def __pow__(self, power):
        if not isinstance(power, Tensor):
            power = Tensor(power, autograd=False, device=self.device)
        op: Op = PowOp(self, power, compute_mode=self.device)
        return op.calc()

    def mm(self, x):
        op: Op = MatMulOp(self, x, compute_mode=self.device)
        return op.calc()

    def exp(self):
        op: Op = ExpOp(self, compute_mode=self.device)
        return op.calc()

    def log(self):
        op: Op = LogOp(self, compute_mode=self.device)
        return op.calc()

    def sin(self):
        op: Op = SinOp(self, compute_mode=self.device)
        return op.calc()

    def cos(self):
        op: Op = CosOp(self, compute_mode=self.device)
        return op.calc()

    def sigmoid(self):
        op: Op = SigmoidOp(self, compute_mode=self.device)
        return op.calc()

    def tanh(self):
        op: Op = TanhOp(self, compute_mode=self.device)
        return op.calc()

    def relu(self):
        op: Op = ReLuOp(self, compute_mode=self.device)
        return op.calc()
    
    def leakyrelu(self, alpha = 0.01):
        op: Op = LeakyReLuOp(self, alpha, compute_mode=self.device)
        return op.calc()

    def softmax(self, axis = -1):
        op: Op = SoftmaxOp(self, axis = axis, compute_mode=self.device)
        return op.calc()

    def abs(self):
        op: Op = AbsOp(self, compute_mode=self.device)
        return op.calc()

    def sum(self, axes):
        op: Op = SumOp(self, axes, compute_mode=self.device)
        return op.calc()

    def max(self, axes):
        op: Op = MaxOp(self, axes, compute_mode=self.device)
        return op.calc()

    def mean(self, axes):
        op: Op = MeanOp(self, axes, compute_mode=self.device)
        return op.calc()

    def var(self, axes):
        op: Op = VarOp(self, axes, compute_mode=self.device)
        return op.calc()

    def sqrt(self):
        op: Op = SqrtOp(self, compute_mode=self.device)
        return op.calc()
    
    def concat(self, others, axis):
        op: Op = ConcatOp(self, others, axis, compute_mode=self.device)
        return op.calc()
    
    def cls(self):
        op: Op = ClsOp(self, compute_mode=self.device)
        return op.calc()

    def broadcast(self, other):
        # 如果是tuple
        if isinstance(other,tuple):
            sp = other
        else:
            sp = other.shape
        if self.shape == sp:
            return self, other

        s1 = list(self.shape)
        s2 = list(sp)
        if len(s1) > len(s2):
            s2 = [1] * (len(s1) - len(s2)) + s2
            if s1 == s2:
                t = BroadcastOp(other, self.shape, compute_mode=self.device).calc()
                return t, other
        else:
            s1 = [1] * (len(s2) - len(s1)) + s1
            if s1 == s2:
                t = BroadcastOp(self, sp, compute_mode=self.device).calc()
                return self, t

        s = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                if s1[i] == 1:
                    s.append(s2[i])
                elif s2[i] == 1:
                    s.append(s1[i])
                else:
                    raise Exception("cannot broadcast")
            else:
                s.append(s1[i])

        if s != list(self.shape):
            t1 = BroadcastOp(self, s, compute_mode=self.device).calc()
        else:
            t1 = self

        if s != list(sp):
            t2 = BroadcastOp(other, s, compute_mode=self.device).calc()
        else:
            t2 = other
        return t1, t2

    def squeeze(self, dim):
        op: Op = SqueezeOp(self, dim, compute_mode=self.device)
        return op.calc()

    def unsqueeze(self, dim):
        op: Op = UnsqueezeOp(self, dim, compute_mode=self.device)
        return op.calc()

    def transpose(self, axes: Iterable[int] = None):
        op: Op = TransposeOp(self, axes, compute_mode=self.device)
        return op.calc()

    def reshape(self, shape):
        op: Op = ReshapeOp(self, shape, compute_mode=self.device)
        return op.calc()

    def rearrange(self, pattern, **axes_lengths):
        op: Op = RearrangeOp(self, pattern, compute_mode=self.device, **axes_lengths)
        return op.calc()

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class TcGraph:
    instance = None

    def __init__(self):
        self.tmap = dict()
        # (op_name, (input1, input2, ...), (output1, output2, ...))
        self.graph = list()

    @classmethod
    def get_instantce(cls):
        if not cls.instance:
            cls.instance = TcGraph()

        return cls.instance

    @classmethod
    def GetTensor(cls, t):
        return cls.get_instantce().getTensor(t)

    @classmethod
    def Compile(cls):
        return cls.get_instantce().compile()

    @classmethod
    def Clear(cls):
        return cls.get_instantce().clear()

    def compile(self):
        '''
        Convert TcGraph into T-Lang program.
        '''

        graph = self.graph
        tensor_dict = dict(map(reversed, self.tmap.items()))

        op_list = ['def main(){\n']

        tensor_input = set()
        tensor_mid = set()

        for (_, __, out) in graph:
            tensor_mid.update(out)

        for (_, inp, __) in graph:
            tensor_input.update(set(inp).difference(tensor_mid))

        # create all input tensor
        for id in tensor_input:
            t = tensor_dict[id]
            shape = 'x'.join(str(d) for d in t.data.shape)
            data = ', '.join(str(e) for e in t.data.flat)
            op = f'  var v{id}<{shape}> = [{data}];\n'
            op_list.append(op)

        while True:
            # if graph not empty, find all which input all generated.
            is_emitable = False
            for (name, inp, out) in graph:
                assert len(out) == 1 and "for now only support 1 result."
                out = out[0]

                is_emitable = out not in tensor_input and set(inp).issubset(tensor_input)
                params = ', '.join(f'v{tid}' for tid in inp)

                if is_emitable:
                    op = f'  var v{out} = {name}({params});\n'
                    if name in ['add', 'matmul']:
                        assert len(inp) == 2 and 'binop must have 2 op.'
                        # TODO: Add more.
                        binop_dict = {
                            'add': '+',
                            'matmul': '.',
                        }
                        op = f'  var v{out} = v{inp[0]} {binop_dict[name]} v{inp[1]};\n'

                    op_list.append(op)

                    tensor_input.add(out)
                    tensor_mid.remove(out)

                    # if cur op's result is the last, also emit printOp.
                    if len(tensor_mid) == 0:
                        op_list.append(f'  print(v{out});\n')
            if not is_emitable:
                break

        op_list.append('}\n')
        return ''.join(op_list)

    def clear(self):
        self.tmap.clear()
        self.graph.clear()

    def getTensor(self, t):
        '''
        return tensor internal repr id.
        '''
        tmap = self.tmap
        assert type(t) == Tensor and "getTensor input only suppor Tensor."
        if t not in tmap:
            # print('TcGraph: Warning: current tensor not managed.')
            return self.addTensor(t)

        # assert t.get_tcg_id() == tmap[t] and "tcg_id and id managed in TcGraph must be the same."
        return tmap[t]

    @classmethod
    def AddTensor(cls, t):
        return cls.get_instantce().addTensor(t)

    def addTensor(self, t):
        '''
        alloc a internal repr id for given tensor.
        '''
        tmap = self.tmap
        if t not in tmap:
            tmap[t] = len(tmap)
        return self.getTensor(t)

    @classmethod
    def AddOp(cls, op_name, inputs, outputs):
        return cls.get_instantce().addOp(op_name, inputs, outputs)

    def addOp(self, op_name, inputs, outputs):
        self.graph.append((op_name,
                           tuple(self.getTensor(i) for i in inputs),
                           tuple(self.addTensor(o) for o in outputs)
                           ))


# TODO grad_fn 是否支持静态、动态重载
class Op:

    def __init__(self, args, tcc_opname='unsupported', compute_mode="cpu"):
        self.input: List[Tensor] = args
        self.output: [Tensor, None] = None
        self.grad_fn = []
        self.grad_fn_gpu = []
        self.compute_mode = compute_mode

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        raise NotImplementedError

    def calc_gpu(self):
        raise NotImplementedError

    def backward(self, grad: np.ndarray):
        assert len(self.input) == len(self.grad_fn)

        for i in range(len(self.input)):
            if self.input[i].autograd:
                if self.compute_mode == "cpu":
                    self.input[i].backward(self.grad_fn[i](grad, self.output, self.input), id(self.output))
                else:
                    # print(i, grad.shape, self.output.shape, self.input[i].shape)
                    self.input[i].backward(self.grad_fn_gpu[i](grad, self.output, self.input), id(self.output))

    def add_dependency(self):
        for i in range(len(self.input)):
            output_id = id(self.output)
            if id(self.output) not in self.input[i].dependents:
                self.input[i].dependents[output_id] = 1
            else:
                self.input[i].dependents[output_id] += 1


class AddOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor, compute_mode="cpu"):
        t1, t2 = t1.broadcast(t2)
        super(AddOp, self).__init__([t1, t2], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data),
            lambda grad, out, args: grad * np.ones_like(args[1].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * cp.ones_like(args[0].data),
            lambda grad, out, args: grad * cp.ones_like(args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc_gpu(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data + self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('add', [self.input[0], self.input[1]], [self.output])
        return self.output

    def calc(self) -> Tensor:
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data + self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('add', [self.input[0], self.input[1]], [self.output])
        return self.output


class SubOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor, compute_mode="cpu"):
        t1, t2 = t1.broadcast(t2)
        super(SubOp, self).__init__([t1, t2], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data),
            lambda grad, out, args: grad * -np.ones_like(args[1].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * cp.ones_like(args[0].data),
            lambda grad, out, args: grad * -cp.ones_like(args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc_gpu(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data - self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('sub', [self.input[0], self.input[1]], [self.output])
        return self.output

    def calc(self) -> Tensor:
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data - self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('sub', [self.input[0], self.input[1]], [self.output])
        return self.output


class MulOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor, compute_mode="cpu"):
        t1, t2 = t1.broadcast(t2)
        super(MulOp, self).__init__([t1, t2], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * args[1].data,
            lambda grad, out, args: grad * args[0].data
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * args[1].data,
            lambda grad, out, args: grad * args[0].data
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data * self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('mul', [self.input[0], self.input[1]], [self.output])
        return self.output

    def calc_gpu(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data * self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('mul', [self.input[0], self.input[1]], [self.output])
        return self.output


class DivOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor, compute_mode="cpu"):
        t1, t2 = t1.broadcast(t2)
        super(DivOp, self).__init__([t1, t2], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad / args[1].data,
            lambda grad, out, args: grad * -args[0].data / (args[1].data * args[1].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad / args[1].data,
            lambda grad, out, args: grad * -args[0].data / (args[1].data * args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data / self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('div', [self.input[0], self.input[1]], [self.output])
        return self.output

    def calc_gpu(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data / self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('div', [self.input[0], self.input[1]], [self.output])
        return self.output


class PowOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor, compute_mode="cpu"):
        t1, t2 = t1.broadcast(t2)
        super(PowOp, self).__init__([t1, t2], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * args[1].data * np.power(args[0].data, args[1].data - 1),
            lambda grad, out, args: grad * np.log(args[0].data) * np.power(args[0].data, args[1].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * args[1].data * cp.power(args[0].data, args[1].data - 1),
            lambda grad, out, args: grad * cp.log(args[0].data) * cp.power(args[0].data, args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.power(self.input[0].data, self.input[1].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('pow', [self.input[0], self.input[1]], [self.output])
        return self.output

    def calc_gpu(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(cp.power(self.input[0].data, self.input[1].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('pow', [self.input[0], self.input[1]], [self.output])
        return self.output


class NegOp(Op):

    def __init__(self, t1: Tensor, compute_mode="cpu"):
        super(NegOp, self).__init__([t1], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * -np.ones_like(args[0].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * -cp.ones_like(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(-self.input[0].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('neg', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(-self.input[0].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('neg', [self.input[0]], [self.output])
        return self.output

class MatMulOp(Op):
    def __init__(self, t1: Tensor, t2: Tensor, compute_mode="cpu"):
        super(MatMulOp, self).__init__([t1, t2], compute_mode=compute_mode)
    
        self.grad_fn = [
            lambda grad, out, args: grad @ args[1].data.transpose(
                list(range(len(args[1].data.shape) - 2)) + [len(args[1].data.shape) - 1, len(args[1].data.shape) - 2]),
            lambda grad, out, args: args[0].data.transpose(
                list(range(len(args[0].data.shape) - 2)) + [len(args[0].data.shape) - 1, len(args[0].data.shape) - 2]) @ grad
        ]

        self.grad_fn_gpu = [
            lambda grad, out, args: grad @ args[1].data.transpose(
                list(range(len(args[1].data.shape) - 2)) + [len(args[1].data.shape) - 1, len(args[1].data.shape) - 2]),
            lambda grad, out, args: args[0].data.transpose(
                list(range(len(args[0].data.shape) - 2)) + [len(args[0].data.shape) - 1, len(args[0].data.shape) - 2]) @ grad
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data @ (self.input[1].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('matmul', [self.input[0], self.input[1]], [self.output])
        return self.output

    def calc_gpu(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data @ (self.input[1].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('matmul', [self.input[0], self.input[1]], [self.output])
        return self.output


class ExpOp(Op):
    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(ExpOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * out.data
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * out.data
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.exp(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('exp', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.exp(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('exp', [self.input[0]], [self.output])
        return self.output


class LogOp(Op):
    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(LogOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad / args[0].data
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad / args[0].data
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.log(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('log', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.log(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('log', [self.input[0]], [self.output])
        return self.output


class SinOp(Op):
    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(SinOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * np.cos(args[0].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * cp.cos(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.sin(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('sin', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.sin(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('sin', [self.input[0]], [self.output])
        return self.output


class CosOp(Op):
    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(CosOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * -np.sin(args[0].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * -cp.sin(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.cos(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('cos', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.cos(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('cos', [self.input[0]], [self.output])
        return self.output


class SigmoidOp(Op):

    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(SigmoidOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * out.data * (1 - out.data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * out.data * (1 - out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(1 / (1 + np.exp(-self.input[0].data)), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('sigmoid', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(1 / (1 + cp.exp(-self.input[0].data)), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('sigmoid', [self.input[0]], [self.output])
        return self.output


class TanhOp(Op):

    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(TanhOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * (1 - out.data * out.data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * (1 - out.data * out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.tanh(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('tanh', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.tanh(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('tanh', [self.input[0]], [self.output])
        return self.output


class ReLuOp(Op):
    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(ReLuOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * (args[0].data > 0)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * (args[0].data > 0)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.maximum(self.input[0].data, 0), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('relu', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.maximum(self.input[0].data, 0), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('relu', [self.input[0]], [self.output])
        return self.output


class LeakyReLuOp(Op):
    def __init__(self, t: Tensor, alpha: float = 0.01, compute_mode="cpu"):
        super(LeakyReLuOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * (args[0].data > 0) + grad * (args[0].data < 0) * alpha
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * (args[0].data > 0) + grad * (args[0].data < 0) * alpha
        ]
        self.alpha = alpha
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        alpha = self.alpha
        if self.output is None:
            self.output: Tensor = Tensor(np.maximum(self.input[0].data, 0) + alpha * np.minimum(self.input[0].data, 0),
                                         creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp(f'leakyrelu_{alpha}', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        alpha = self.alpha
        if self.output is None:
            self.output: Tensor = Tensor(cp.maximum(self.input[0].data, 0) + alpha * cp.minimum(self.input[0].data, 0),
                                         creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp(f'leakyrelu_{alpha}', [self.input[0]], [self.output])
        return self.output
    


class AbsOp(Op):
    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(AbsOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * np.sign(args[0].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * cp.sign(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.abs(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('abs', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.abs(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('abs', [self.input[0]], [self.output])
        return self.output


def reduce_shape(shape: tuple, axes: [None, int, Iterable]):
    if axes is None:
        return None, (1,) * len(shape)

    _shape = list(shape)
    if isinstance(axes, int):
        axes = [axes]
    else:
        axes = list(axes)

    for i in range(len(axes)):
        if axes[i] < 0:
            axes[i] += len(shape)
        _shape[axes[i]] = 1

    axes = tuple(axes)

    _shape = tuple(_shape)
    return axes, _shape


class SumOp(Op):

    def __init__(self, t: Tensor, axes: [int, Iterable], compute_mode="cpu"):
        super(SumOp, self).__init__([t], compute_mode=compute_mode)

        self.axes, self._shape = reduce_shape(t.shape, axes)

        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._shape) * np.ones_like(args[0].data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad.reshape(self._shape) * cp.ones_like(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('sum', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('sum', [self.input[0]], [self.output])
        return self.output


class MaxOp(Op):

    def __init__(self, t: Tensor, axes: int, compute_mode="cpu"):
        super(MaxOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * (args[0].data == out.data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * (args[0].data == out.data)
        ]
        self.axes = axes
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.max(axis=self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('max', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.max(axis=self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('max', [self.input[0]], [self.output])
        return self.output


class MeanOp(Op):

    def __init__(self, t: Tensor, axes: [int, Iterable], compute_mode="cpu"):
        super(MeanOp, self).__init__([t], compute_mode=compute_mode)

        self.axes, self._shape = reduce_shape(t.shape, axes)

        self.N = 1
        for axis in self.axes:
            self.N *= t.shape[axis]

        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._shape) * np.ones_like(args[0].data) / self.N
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad.reshape(self._shape) * cp.ones_like(args[0].data) / self.N
        ]
        self.calc()
        self.add_dependency()

    # TODO: TcGraph: argument config is needed.
    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.axes) / self.N,
                                         creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('mean', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.axes) / self.N,
                                         creation_op=self,
                                         autograd=any(t.autograd for t in self.input),
                                         device=self.compute_mode)
            TcGraph.AddOp('mean', [self.input[0]], [self.output])
        return self.output


class VarOp(Op):

    def __init__(self, t: Tensor, axes: [int, Iterable], compute_mode="cpu"):
        super(VarOp, self).__init__([t], compute_mode=compute_mode)

        self.axes, self._shape = reduce_shape(t.shape, axes)

        self.N = 1
        for axis in self.axes:
            self.N *= t.shape[axis]

        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._shape) *
                                    2 * (args[0].data - args[0].data.sum(self.axes, keepdims=True) / self.N) / self.N
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad.reshape(self._shape) *
                                    2 * (args[0].data - args[0].data.sum(self.axes, keepdims=True) / self.N) / self.N
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            data = self.input[0].data
            mean_val = data.sum(axis=self.axes, keepdims=True) / self.N
            data = data - mean_val
            data = data * data
            self.output: Tensor = Tensor(data.sum(axis=self.axes) / self.N, creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)

        return self.output

    def calc_gpu(self):
        if self.output is None:
            data = self.input[0].data
            mean_val = data.sum(axis=self.axes, keepdims=True) / self.N
            data = data - mean_val
            data = data * data
            self.output: Tensor = Tensor(data.sum(axis=self.axes) / self.N, creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
        return self.output


class SqrtOp(Op):
    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(SqrtOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            lambda grad, out, args: grad * 0.5 * (1 / out.data)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad * 0.5 * (1 / out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.sqrt(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.sqrt(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
        return self.output

class ConcatOp(Op):
    def __init__(self, t1: Tensor, t2: Tensor, axis: int, compute_mode="cpu"):
        super(ConcatOp, self).__init__([t1, t2], compute_mode=compute_mode)
        self.axis = axis

        self.grad_fn = [
            lambda grad, out, args: np.split(grad, [args[0].data.shape[self.axis]], axis=self.axis)[0],
            lambda grad, out, args: np.split(grad, [args[0].data.shape[self.axis]], axis=self.axis)[1] 
        ]

        self.grad_fn_gpu = [
            lambda grad, out, args: np.split(grad, [args[0].data.shape[self.axis]], axis=self.axis)[0],
            lambda grad, out, args: np.split(grad, [args[0].data.shape[self.axis]], axis=self.axis)[1] 
        ]

        self.calc()
        self.add_dependency()
        
    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.concatenate((self.input[0].data, self.input[1].data), axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
        return self.output
    
    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.concatenate((self.input[0].data, self.input[1].data), axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)

        return self.output

class ClsOp(Op):
    def __init__(self, t: Tensor, compute_mode="cpu"):
        super(ClsOp, self).__init__([t], compute_mode=compute_mode)
        self.grad_fn = [
            # 只需要一维，其他的为0
            lambda grad, out, args: np.concatenate((grad, np.zeros_like(args[0].data)[:, 1:, ...]), axis=1)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: cp.concatenate((grad, cp.zeros_like(args[0].data)[:, 1:, ...]), axis=1)
        ]
        self.calc()
        self.add_dependency()
        
    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output = Tensor(self.input[0].data[:, :1], creation_op=self, autograd=any(t.autograd for t in self.input), device=self.compute_mode)
        return self.output
    
    def calc_gpu(self):
        if self.output is None:
            self.output = Tensor(self.input[0].data[:, :1], creation_op=self, autograd=any(t.autograd for t in self.input), device=self.compute_mode)
        return self.output


class SoftmaxOp(Op):
    def __init__(self, t: Tensor, axis=-1, compute_mode="cpu"):
        super(SoftmaxOp, self).__init__([t], compute_mode=compute_mode)
        self.axis = axis
        self.grad_fn = [
            lambda grad, out, args: out.data * (grad - np.sum(grad * out.data, axis=self.axis, keepdims=True))
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: out.data * (grad - cp.sum(grad * out.data, axis=self.axis, keepdims=True))
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.compute_mode != "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(scipy.special.softmax(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('softmax', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            exp_data = cp.exp(self.input[0].data - cp.max(self.input[0].data, axis=self.axis, keepdims=True))
            self.output: Tensor = Tensor(exp_data / cp.sum(exp_data, axis=self.axis, keepdims=True), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('softmax', [self.input[0]], [self.output])
        return self.output



class BroadcastOp(Op):
    def __init__(self, t: Tensor, shape: [int], compute_mode="cpu"):
        super(BroadcastOp, self).__init__([t], compute_mode=compute_mode)
        self.shape = shape
        self.axes = []
        if len(shape) > len(t.shape):
            self.axes = list(range(len(shape) - len(t.shape)))

        offset = len(shape) - len(t.shape)
        for i in range(len(t.shape)):
            if t.shape[i] != shape[i + offset]:
                self.axes.append(i + offset)

        self.axes = tuple(self.axes)
        
        self.grad_fn = [
            lambda grad, out, args: grad.sum(axis=self.axes).reshape(args[0].shape)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad.sum(axis=self.axes).reshape(args[0].shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.broadcast_to(self.input[0].data, self.shape), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('broadcast', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            
            self.output: Tensor = Tensor(cp.broadcast_to(self.input[0].data, self.shape), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('broadcast', [self.input[0]], [self.output])
        return self.output


class SqueezeOp(Op):
    def __init__(self, t: Tensor, axis: int, compute_mode="cpu"):
        super(SqueezeOp, self).__init__([t], compute_mode=compute_mode)
        self.axis = axis
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.squeeze(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('squeeze', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.squeeze(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('squeeze', [self.input[0]], [self.output])
        return self.output


class UnsqueezeOp(Op):
    def __init__(self, t: Tensor, axis: int, compute_mode="cpu"):
        super(UnsqueezeOp, self).__init__([t], compute_mode=compute_mode)
        self.axis = axis
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(np.expand_dims(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('unsqueeze', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(cp.expand_dims(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('unsqueeze', [self.input[0]], [self.output])
        return self.output


class TransposeOp(Op):

    def __init__(self, t: Tensor, axes: Iterable[int] = None, compute_mode="cpu"):
        super(TransposeOp, self).__init__([t], compute_mode=compute_mode)
        if axes is None:
            self.axes = list(range(len(t.shape) - 1, -1, -1))
            
        else:
            self.axes = axes

        self.grad_fn = [
            lambda grad, out, args: grad.transpose(
                # list(range(len(self.axes))).sort(key=lambda x: self.axes[x])
                sorted(list(range(len(self.axes))), key=lambda x: self.axes[x])
            )
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad.transpose(
                # list(range(len(self.axes))).sort(key=lambda x: self.axes[x])
                sorted(list(range(len(self.axes))), key=lambda x: self.axes[x])
            )
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.transpose(self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('transpose', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.transpose(self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('transpose', [self.input[0]], [self.output])
        return self.output


class ReshapeOp(Op):
    def __init__(self, t: Tensor, shape: [int], compute_mode="cpu"):
        super(ReshapeOp, self).__init__([t], compute_mode=compute_mode)

        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = int(np.prod(t.shape) / np.prod(shape))
                break
        shape = tuple(shape)

        self.shape = shape
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.reshape(self.shape), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('reshape', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.reshape(self.shape), creation_op=self,
                                         autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('reshape', [self.input[0]], [self.output])
        return self.output



class RearrangeOp(Op):
    def __init__(self, t: Tensor, pattern: str, compute_mode="cpu", **axes_lengths):
        super(RearrangeOp, self).__init__([t], compute_mode=compute_mode)

        self.pattern = pattern
        self.inverse_pattern = pattern.split('->')[1] + '->' + pattern.split('->')[0]
        self.grad_fn = [
            lambda grad, out, args: einops_rearrange(grad, self.inverse_pattern, **axes_lengths)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: einops_rearrange(grad, self.inverse_pattern, **axes_lengths)
        ]
        self.calc(**axes_lengths)
        self.add_dependency()

    def calc(self, **axes_lengths):
        if not self.compute_mode == "cpu":
            return self.calc_gpu(**axes_lengths)
        if self.output is None:
            self.output = Tensor(einops_rearrange(self.input[0].data, self.pattern, **axes_lengths), creation_op=self,
                                 autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('rearrange', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self, **axes_lengths):
        if self.output is None:
            self.output = Tensor(einops_rearrange(self.input[0].data, self.pattern, **axes_lengths), creation_op=self,
                                 autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('rearrange', [self.input[0]], [self.output])
        return self.output


class Conv2dOp(Op):
    def __init__(self, t: Tensor, kernel: Tensor, stride: int, padding: int, compute_mode="cpu"):
        super(Conv2dOp, self).__init__([t, kernel], compute_mode=compute_mode)
        self.stride = stride
        self.padding = padding
        self.grad_fn = [
            lambda grad, out, args: self.calc_grad_input(grad),
            lambda grad, out, args: self.calc_grad_kernel(grad)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: self.calc_grad_input_gpu(grad),
            lambda grad, out, args: self.calc_grad_kernel_gpu(grad)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.img_cols, data = self._conv2d(self.input[0].data, self.input[1].data)
            self.output: Tensor = Tensor(data, creation_op=self, autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('conv2d', [self.input[0], self.input[1]], [self.output])
        return self.output
    
    def calc_gpu(self):
        if self.output is None:
            self.img_cols, data = self._conv2d_gpu(self.input[0].data, self.input[1].data)
            self.output: Tensor = Tensor(data, creation_op=self, autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('conv2d', [self.input[0], self.input[1]], [self.output])
        return self.output

    def calc_grad_kernel(self, grad_output: np.ndarray):
        B, _, _, _ = grad_output.shape
        # kernel 的 grad
        YC_grad = self._y_to_YC(grad_output)
        XC_T = self.img_cols.transpose()
        weight_grad = np.dot(XC_T, YC_grad)
        return weight_grad.transpose().reshape(self.input[1].data.shape)
    
    def calc_grad_kernel_gpu(self, grad_output: cp.ndarray):
        B, _, _, _ = grad_output.shape
        # kernel 的 grad
        YC_grad = self._y_to_YC(grad_output)
        XC_T = self.img_cols.transpose()
        weight_grad = cp.dot(XC_T, YC_grad)
        return weight_grad.transpose().reshape(self.input[1].data.shape)

    def calc_grad_input(self, grad_output: np.ndarray):
        # input 的 grad
        grad_output_padding = self._stride_and_padding(grad_output,
                                                       stride=self.stride,
                                                       padding=self.input[1].data.shape[-1] - 1)
        kernel_ = self._rotate180(self.input[1].data)
        result = self._conv2d(grad_output_padding, kernel_, is_grad=True)[1]
        if self.padding > 0:
            if self.stride == 1 or (self.input[0].data.shape[-1] + self.padding * 2 + 1) % self.stride == 0:
                return result[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                add_right_down = (self.input[0].data.shape[-1] + self.padding * 2 + 1) % self.stride
                result = self._stride_and_padding(result,
                                             stride=1,
                                             padding=add_right_down)
                return result[:, :, add_right_down+self.padding:-self.padding, add_right_down+self.padding:-self.padding]

        else:
            if self.stride == 1 or (self.input[0].data.shape[-1] + 1) % self.stride == 0:
                return result
            else:
                add_right_down = (self.input[0].data.shape[-1] + 1) % self.stride
                result = self._stride_and_padding(result,
                                                      stride=1,
                                                      padding=add_right_down)
                return result[:, :, add_right_down:, add_right_down:]
    
    def calc_grad_input_gpu(self, grad_output: cp.ndarray):
        # input 的 grad
        grad_output_padding = self._stride_and_padding_gpu(grad_output,
                                                       stride=self.stride,
                                                       padding=self.input[1].data.shape[-1] - 1)
        kernel_ = self._rotate180_gpu(self.input[1].data)
        result = self._conv2d_gpu(grad_output_padding, kernel_, is_grad=True)[1]
        if self.padding > 0:
            if self.stride == 1 or (self.input[0].data.shape[-1] + self.padding * 2 + 1) % self.stride == 0:
                return result[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                add_right_down = (self.input[0].data.shape[-1] + self.padding * 2 + 1) % self.stride
                result = self._stride_and_padding_gpu(result,
                                             stride=1,
                                             padding=add_right_down)
                return result[:, :, add_right_down+self.padding:-self.padding, add_right_down+self.padding:-self.padding]

        else:
            if self.stride == 1 or (self.input[0].data.shape[-1] + 1) % self.stride == 0:
                return result
            else:
                add_right_down = (self.input[0].data.shape[-1] + 1) % self.stride
                result = self._stride_and_padding_gpu(result,
                                                      stride=1,
                                                      padding=add_right_down)
                return result[:, :, add_right_down:, add_right_down:]

    def _conv2d(self, img: np.ndarray, kernel: np.ndarray, is_grad=False):
        if not is_grad:
            img = np.lib.pad(img, ((0, 0),
                                (0, 0),
                                (self.padding, self.padding),
                                (self.padding, self.padding)), "constant",
                            constant_values=0)
        B, _, H, W = img.shape
        O, I, K, K = kernel.shape
        if not is_grad:
            img_cols = self._img2col(img, kernel_size=K, stride=self.stride)
            kernel_ = kernel.reshape(O, -1).T
            y = np.dot(img_cols, kernel_)  # 矩阵相乘
            output_H, output_W = (H - K) // self.stride + 1, (W - K) // self.stride + 1
            result = self._YC_to_y(y, B, O, output_H, output_W)  # reshape变换输出的形式
        else:
            img_cols = self._img2col(img, kernel_size=K, stride=1)
            kernel_ = kernel.reshape(O, -1).T
            y = np.dot(img_cols, kernel_)  # 矩阵相乘
            output_H, output_W = (H - K) // 1 + 1, (W - K) // 1 + 1
            result = self._YC_to_y(y, B, O, output_H, output_W)  # reshape变换输出的形式
        return img_cols, result
    
    def _conv2d_gpu(self, img: cp.ndarray, kernel: cp.ndarray, is_grad=False):
        if not is_grad:
            img = cp.pad(img, ((0, 0),
                                (0, 0),
                                (self.padding, self.padding),
                                (self.padding, self.padding)), "constant",
                            constant_values=0)
        B, _, H, W = img.shape
        O, I, K, K = kernel.shape
        if not is_grad:
            img_cols = self._img2col_gpu(img, kernel_size=K, stride=self.stride)
            kernel_ = kernel.reshape(O, -1).T
            y = cp.dot(img_cols, kernel_)  # 矩阵相乘
            output_H, output_W = (H - K) // self.stride + 1, (W - K) // self.stride + 1
            result = self._YC_to_y(y, B, O, output_H, output_W)  # reshape变换输出的形式
        else:
            img_cols = self._img2col_gpu(img, kernel_size=K, stride=1)
            kernel_ = kernel.reshape(O, -1).T
            y = cp.dot(img_cols, kernel_)  # 矩阵相乘
            output_H, output_W = (H - K) // 1 + 1, (W - K) // 1 + 1
            result = self._YC_to_y(y, B, O, output_H, output_W)  # reshape变换输出的形式
        return img_cols, result

    # YC变换成Y，前向过程需要
    def _YC_to_y(self, YC, batch_size, channel, output_H, output_W):
        result = YC.reshape((batch_size, YC.shape[0] // batch_size, -1)).reshape(
            (batch_size, output_H, output_W, channel))
        return result.transpose((0, 3, 1, 2))

    # y变换成YC，反向传播过程需要
    def _y_to_YC(self, y):
        B, C, H, W = y.shape
        result = y.transpose((0, 2, 3, 1)).reshape(B * W * H, -1)
        return result

    def _img2col(self, img: np.ndarray, kernel_size, stride=1):
        B, C, H, W = img.shape
        out_h = (H - kernel_size) // stride + 1
        out_w = (W - kernel_size) // stride + 1

        col = np.zeros((B, C, kernel_size, kernel_size, out_h, out_w))

        for y in range(kernel_size):
            y_max = y + stride * out_h
            for x in range(kernel_size):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = np.ascontiguousarray(col.transpose((0, 4, 5, 1, 2, 3))).reshape(B * out_h * out_w, -1)
        # (B * out_h * out_w, C * kernel_size * kernel_size)
        return col

    def _img2col_gpu(self, img: cp.ndarray, kernel_size, stride=1):
        B, C, H, W = img.shape
        out_h = (H - kernel_size) // stride + 1
        out_w = (W - kernel_size) // stride + 1

        col = cp.zeros((B, C, kernel_size, kernel_size, out_h, out_w))

        for y in range(kernel_size):
            y_max = y + stride * out_h
            for x in range(kernel_size):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = cp.ascontiguousarray(col.transpose((0, 4, 5, 1, 2, 3))).reshape(B * out_h * out_w, -1)
        # (B * out_h * out_w, C * kernel_size * kernel_size)
        return col
    
    def _stride_and_padding(self, grad_output, stride, padding):
        if stride > 1:
            N, O, output_H, output_W = grad_output.shape
            inserted_H, inserted_W = output_H + (output_H - 1) * (stride - 1), output_W + (output_W - 1) * (stride - 1)
            inserted_eta = np.zeros((N, O, inserted_H, inserted_W))
            inserted_eta[:, :, ::stride, ::stride] = grad_output
            grad_output = inserted_eta
        grad_output = np.lib.pad(grad_output, ((0, 0),
                                               (0, 0),
                                               (padding, padding),
                                               (padding, padding)), "constant",
                                 constant_values=0)
        return grad_output
    
    def _stride_and_padding_gpu(self, grad_output, stride, padding):
        if stride > 1:
            N, O, output_H, output_W = grad_output.shape
            inserted_H, inserted_W = output_H + (output_H - 1) * (stride - 1), output_W + (output_W - 1) * (stride - 1)
            inserted_eta = cp.zeros((N, O, inserted_H, inserted_W))
            inserted_eta[:, :, ::stride, ::stride] = grad_output
            grad_output = inserted_eta
        grad_output = cp.pad(grad_output, ((0, 0),
                                            (0, 0),
                                            (padding, padding),
                                            (padding, padding)), "constant",
                                constant_values=0)
        return grad_output

    def _rotate180(self, kernel):
        # 旋转90+90度构成旋转180度
        # weight = np.rot90(weight, axes=(2, 3))
        # weight = np.rot90(weight, axes=(2, 3))
        _, C, _, _ = kernel.shape
        weight_flip = np.flip(kernel, (2, 3))  # 卷积核旋转180度
        weight_flip_swap = np.swapaxes(weight_flip, 0, 1)  # 交换输入、输出通道的顺序[C,O,H,W]
        return weight_flip_swap
    
    def _rotate180_gpu(self, kernel):
        # 旋转90+90度构成旋转180度
        # weight = np.rot90(weight, axes=(2, 3))
        # weight = np.rot90(weight, axes=(2, 3))
        _, C, _, _ = kernel.shape
        weight_flip = cp.flip(kernel, (2, 3))  # 卷积核旋转180度
        weight_flip_swap = cp.swapaxes(weight_flip, 0, 1)  # 交换输入、输出通道的顺序[C,O,H,W]
        return weight_flip_swap


class MaxPool2dOp(Op):
    def __init__(self, t: Tensor, kernel_size: int, stride: int, padding: int, ceil_mode: bool, compute_mode="cpu"):
        super(MaxPool2dOp, self).__init__([t], compute_mode=compute_mode)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.grad_fn = [
            lambda grad, out, args: self.max_pool2d_grad_input(grad, args[0].data, out.data,
                                                               self.kernel_size, self.stride,
                                                               self.padding)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: self.max_pool2d_grad_input_gpu(grad, args[0].data, out.data,
                                                               self.kernel_size, self.stride,
                                                               self.padding)
        ]
        self.calc()
        self.add_dependency()

    def max_pool2d_grad_input(self, grad: np.ndarray, input: np.ndarray, output: np.ndarray, kernel_size: int,
                              stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        grad_input = np.zeros(input.shape)
        # for b in range(batch_size):
        #     for c in range(in_channels):
        #         for h in range(out_height):
        #             for w in range(out_width):
        #                 grad_input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
        #                     grad[b, c, h, w] * (
        #                             input[b, c, h * stride:h * stride + kernel_size,
        #                             w * stride:w * stride + kernel_size] ==
        #                             output[b, c, h, w])
        for h in range(out_height):
            for w in range(out_width):
                grad_input[:, :, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
                    grad[:, :, h, w].reshape(batch_size, in_channels, 1, 1) * (
                            input[:, :, h * stride:h * stride + kernel_size,
                            w * stride:w * stride + kernel_size] ==
                            output[:, :, h, w].reshape(batch_size, in_channels, 1, 1))

        return grad_input
    
    def max_pool2d_grad_input_gpu(self, grad: cp.ndarray, input: cp.ndarray, output: cp.ndarray, kernel_size: int,
                              stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        grad_input = cp.zeros(input.shape)

        for h in range(out_height):
            for w in range(out_width):
                grad_input[:, :, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
                    grad[:, :, h, w].reshape(batch_size, in_channels, 1, 1) * (
                            input[:, :, h * stride:h * stride + kernel_size,
                            w * stride:w * stride + kernel_size] ==
                            output[:, :, h, w].reshape(batch_size, in_channels, 1, 1))
        return grad_input

    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(
                self.max_pool2d(self.input[0].data, self.kernel_size, self.stride, self.padding), creation_op=self,
                autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('max_pool2d', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(
                self.max_pool2d_gpu(self.input[0].data, self.kernel_size, self.stride, self.padding), creation_op=self,
                autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('max_pool2d', [self.input[0]], [self.output])
        return self.output
    
    def max_pool2d(self, input: np.ndarray, kernel_size: int, stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        input = np.lib.pad(input, ((0, 0),
                                    (0, 0),
                                    (padding, padding),
                                    (padding, padding)), "constant",
                            constant_values=0)
        if self.ceil_mode:
            pad_right = (input.shape[3] - kernel_size) % stride # 右边补0的个数
            pad_bottom = (input.shape[2] - kernel_size) % stride # 下边补0的个数
            input = np.lib.pad(input, ((0, 0),
                                        (0, 0),
                                        (0, pad_bottom),
                                        (0, pad_right)), "constant",
                                constant_values=0)
            batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        output = np.zeros((batch_size, in_channels, out_height, out_width))
        # for b in range(batch_size):
        #     for c in range(in_channels):
        #         for h in range(out_height):
        #             for w in range(out_width):
        #                 output[b, c, h, w] = np.max(
        #                     input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size])
        for h in range(out_height):
            for w in range(out_width):
                output[:, :, h, w] = np.max(
                    input[:, :, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size], axis=(2, 3))
        return output

    def max_pool2d_gpu(self, input: cp.ndarray, kernel_size: int, stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        input = cp.pad(input, ((0, 0),
                                (0, 0),
                                (padding, padding),
                                (padding, padding)), "constant",
                        constant_values=0)
        if self.ceil_mode:
            pad_right = (input.shape[3] - kernel_size) % stride # 右边补0的个数
            pad_bottom = (input.shape[2] - kernel_size) % stride # 下边补0的个数
            input = cp.pad(input, ((0, 0),
                                    (0, 0),
                                    (0, pad_bottom),
                                    (0, pad_right)), "constant",
                            constant_values=0)
            batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        output = cp.zeros((batch_size, in_channels, out_height, out_width))
        
        for h in range(out_height):
            for w in range(out_width):
                output[:, :, h, w] = cp.max(
                    input[:, :, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size], axis=(2, 3))
        return output
    
class AvgPool2dOp(Op):
    def __init__(self, t: Tensor, kernel_size: int, stride: int, padding: int, ceil_mode: bool, compute_mode="cpu"):
        super(AvgPool2dOp, self).__init__([t], compute_mode=compute_mode)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.grad_fn = [
            lambda grad, out, args: self.avg_pool2d_grad_input(grad, args[0].data, out.data,
                                                               self.kernel_size, self.stride,
                                                               self.padding)
        ]
        self.grad_fn_gpu = [
            lambda grad, out, args: self.avg_pool2d_grad_input_gpu(grad, args[0].data, out.data,
                                                               self.kernel_size, self.stride,
                                                               self.padding)
        ]
        self.calc()
        self.add_dependency()

    def avg_pool2d_grad_input(self, grad: np.ndarray, input: np.ndarray, output: np.ndarray, kernel_size: int,
                              stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        grad_input = np.zeros(input.shape)
        # for b in range(batch_size):
        #     for c in range(in_channels):
        #         for h in range(out_height):
        #             for w in range(out_width):
        #                 grad_input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
        #                     grad[b, c, h, w]
        for h in range(out_height):
            for w in range(out_width):
                grad_input[:, :, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
                    grad[:, :, h, w].reshape(batch_size, in_channels, 1, 1)
                
        return grad_input / (kernel_size * kernel_size)

    def avg_pool2d_grad_input_gpu(self, grad: cp.ndarray, input: cp.ndarray, output: cp.ndarray, kernel_size: int,
                              stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        grad_input = cp.zeros(input.shape)
        # for b in range(batch_size):
        #     for c in range(in_channels):
        #         for h in range(out_height):
        #             for w in range(out_width):
        #                 grad_input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
        #                     grad[b, c, h, w]
        for h in range(out_height):
            for w in range(out_width):
                grad_input[:, :, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
                    grad[:, :, h, w].reshape(batch_size, in_channels, 1, 1)
                
        return grad_input / (kernel_size * kernel_size)
    
    def calc(self):
        if not self.compute_mode == "cpu":
            return self.calc_gpu()
        if self.output is None:
            self.output: Tensor = Tensor(
                self.avg_pool2d(self.input[0].data, self.kernel_size, self.stride, self.padding), creation_op=self,
                autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('avg_pool2d', [self.input[0]], [self.output])
        return self.output

    def calc_gpu(self):
        if self.output is None:
            self.output: Tensor = Tensor(
                self.avg_pool2d_gpu(self.input[0].data, self.kernel_size, self.stride, self.padding), creation_op=self,
                autograd=any(t.autograd for t in self.input), device=self.compute_mode)
            TcGraph.AddOp('avg_pool2d', [self.input[0]], [self.output])
        return self.output

    def avg_pool2d(self, input: np.ndarray, kernel_size: int, stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        input = np.lib.pad(input, ((0, 0),
                                    (0, 0),
                                    (padding, padding),
                                    (padding, padding)), "constant",
                            constant_values=0)
        if self.ceil_mode:
            pad_right = (input.shape[3] - kernel_size) % stride # 右边补0的个数
            pad_bottom = (input.shape[2] - kernel_size) % stride # 下边补0的个数
            input = np.lib.pad(input, ((0, 0),
                                        (0, 0),
                                        (0, pad_bottom),
                                        (0, pad_right)), "constant",
                                constant_values=0)
            batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        output = np.zeros((batch_size, in_channels, out_height, out_width))
        # for b in range(batch_size):
        #     for c in range(in_channels):
        #         for h in range(out_height):
        #             for w in range(out_width):
        #                 output[b, c, h, w] = np.mean(
        #                     input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size])
        for h in range(out_height):
            for w in range(out_width):
                output[:, :, h, w] = np.mean(
                    input[:, :, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size], axis=(2, 3))
        return output

    def avg_pool2d_gpu(self, input: cp.ndarray, kernel_size: int, stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        input = cp.pad(input, ((0, 0),
                                (0, 0),
                                (padding, padding),
                                (padding, padding)), "constant",
                        constant_values=0)
        if self.ceil_mode:
            pad_right = (input.shape[3] - kernel_size) % stride # 右边补0的个数
            pad_bottom = (input.shape[2] - kernel_size) % stride # 下边补0的个数
            input = cp.pad(input, ((0, 0),
                                    (0, 0),
                                    (0, pad_bottom),
                                    (0, pad_right)), "constant",
                            constant_values=0)
            batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        output = cp.zeros((batch_size, in_channels, out_height, out_width))

        for h in range(out_height):
            for w in range(out_width):
                output[:, :, h, w] = cp.mean(
                    input[:, :, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size], axis=(2, 3))
        return output

def log(t: Tensor) -> Tensor:
    return t.log()


def exp(t: Tensor) -> Tensor:
    return t.exp()


def sin(t: Tensor) -> Tensor:
    return t.sin()


def cos(t: Tensor) -> Tensor:
    return t.cos()


def tanh(t: Tensor) -> Tensor:
    return t.tanh()


def sigmoid(t: Tensor) -> Tensor:
    return t.sigmoid()


def relu(t: Tensor) -> Tensor:
    return t.relu()


def mm(t1: Tensor, t2: Tensor) -> Tensor:
    return t1.mm(t2)


def softmax(t: Tensor, axis: int) -> Tensor:
    return t.softmax(axis)


def abs(t: Tensor) -> Tensor:
    return t.abs()


def sum(t: Tensor, axes: [int, Iterable]) -> Tensor:
    return t.sum(axes)


def max(t: Tensor, dim: int) -> Tensor:
    return t.max(dim)


def mean(t: Tensor, axes: [int, Iterable]) -> Tensor:
    return t.mean(axes)


def var(t: Tensor, axes: [int, Iterable]) -> Tensor:
    return t.var(axes)


def sqrt(t: Tensor) -> Tensor:
    return t.sqrt()


def conv2d(t: Tensor, kernel: Tensor, stride: int, padding: int) -> Tensor:
    return Conv2dOp(t, kernel, stride, padding, compute_mode=t.device).calc()


def max_pool2d(t: Tensor, kernel_size: int, stride: int, padding: int, ceil_mode: bool) -> Tensor:
    return MaxPool2dOp(t, kernel_size, stride, padding, ceil_mode, compute_mode=t.device).calc()


def avg_pool2d(t: Tensor, kernel_size: int, stride: int, padding: int, ceil_mode: bool) -> Tensor:
    return AvgPool2dOp(t, kernel_size, stride, padding, ceil_mode, compute_mode=t.device).calc()
