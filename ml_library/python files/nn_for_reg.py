import numpy as np

class Neuron:
    def __init__(self, data,children = (),_backward = lambda : None):
        self.data = data
        self.grad = 0.0
        self.children = children
        self._backward = lambda : None
    def __mul__(self,other):
        other = other if isinstance(other,Neuron) else Neuron(other)
        out = Neuron(self.data*other.data)
        out.children = (self,other)
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out
    def __add__(self,other):
        other = other if isinstance(other,Neuron) else Neuron(other)
        out = Neuron(self.data + other.data)
        out.children = (self,other)
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
    def __sub__(self,other):
        other = other if isinstance(other,Neuron) else Neuron(other)
        out = Neuron(self.data - other.data)
        out.children = (self,other)
        def _backward():
            self.grad += 1.0*out.grad
            other.grad -= 1.0*out.grad
        out._backward = _backward
        return out
    def relu(self):
        out = Neuron(self.data) if self.data > 0 else Neuron(0)
        out.children = (self,)
        def _backward():
            self.grad += out.grad*1 if self.data > 0 else 0
        out._backward = _backward
        return out
    def __truediv__(self,other):
        other = other if isinstance(other,Neuron) else Neuron(other)
        def _backward():
            self.grad += out.grad/other.data
            other.grad += -(self.data / (other.data ** 2)) * out.grad
        try:
            out = self.data/other.data
            return Neuron(out , children = (self,other),_backward = _backward)
        except ZeroDivisionError:
            out = Neuron(0)
        def _backward():
            self.grad += out.grad/other.data
            other.grad += -(self.data / (other.data ** 2)) * out.grad
        out.children = (self,other)
        out._backward = _backward
        return out
    def __pow__(self,other : float):
        # other = other if isinstance(other,Neuron) else Neuron(other) we are not doing this cuz
        # if other.data < 0 we might have to face a problem of get undefined
        # to avoid that I am considering other over here as an int variable only
        def _backward():
            self.grad += (other*(self.data**(other - 1)))*out.grad
        out = Neuron(self.data**other)
        out.children = (self,)
        out._backward = _backward
        return out
    def __radd__(self,other):
        other = other if isinstance(other,Neuron) else Neuron(other)
        out = Neuron(self.data + other.data)
        out.children = (self,other)
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
    def __rsub__(self,other):
        other = other if isinstance(other,Neuron) else Neuron(other)
        out = Neuron(self.data - other.data)
        out.children = (self,other)
        def _backward():
            self.grad += 1.0*out.grad
            other.grad -= 1.0*out.grad
        out._backward = _backward
        return out
    def __rmul__(self,other):
        other = other if isinstance(other,Neuron) else Neuron(other)
        out = Neuron(self.data * other.data)
        out.children = (self,other)
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out
    def __rtruediv__(self,other):
        other = other if isinstance(other,Neuron) else Neuron(other)
        def _backward():
            self.grad += out.grad/other.data
            other.grad += -(self.data / (other.data ** 2)) * out.grad
        try:
            out = self.data/other.data
            return Neuron(out,children = (self,other),_backward = _backward)
        except ZeroDivisionError:
            out = Neuron(0)
        out.children = (self,other)
        out._backward = _backward
        return out
    def __rpow__(self,other : float):
        out = Neuron(self.data**other)
        def _backward():
            self.grad += (other*(self.data**(other - 1)))*out.grad
        out.children = (self,other)
        out._backward = _backward
        return out
    def __neg__(self):
        out = Neuron(-self.data)
        out.children = (self,)
        def _backward():
            self.grad += -1.0*out.grad
        out._backward = _backward
        return out
    def tanh(self):
        out = Neuron((np.exp(2*self.data) - 1)/(np.exp(2*self.data) + 1))
        out.children = (self,)
        def _backward():
            self.grad += out.grad*(1 - out.data ** 2)
        out._backward = _backward
        return out
    def __repr__(self):
        return f"Neuron({self.data} grad {self.grad})"
    def backward(self):
    # seed gradient at loss
      self.grad = 1.0

      topo = []
      visited = set()
      stack = [(self, 0)]  # (node, state) where state=0 -> first time, state=1 -> after children

      while stack:
          n, state = stack.pop()
          if state == 0:
              if n in visited:
                  continue
              visited.add(n)
              # post-order: push (n,1) then its children with state 0
              stack.append((n, 1))
              for child in n.children:
                  if child not in visited:
                      stack.append((child, 0))
          else:
              # all children processed; now append to topo
              topo.append(n)

      # same as before
      for n in reversed(topo):
          n._backward()
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Node(Module):

    def __init__(self, nin, nonlin=True):
        # intializing the default mode be a relu unit with He activation
        self.w = [Neuron(np.random.normal(0,np.sqrt(2/nin))) for _ in range(nin)]
        self.b = Neuron(0)
        self.nin = nin
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout,weight_intializer = 'He' ,nonlin = False ,**kwargs):
        self.neurons = [Node(nin, nonlin) for _ in range(nout)]
        def He():
            for neuron in self.neurons:
                neuron.w = [Neuron(np.random.normal(0, np.sqrt(2 / neuron.nin))) for _ in range(neuron.nin)]
        def Xavier():
            for neuron in self.neurons:
                neuron.w = [Neuron(np.random.normal(0, np.sqrt(2 / (neuron.nin + nout)))) for _ in range(neuron.nin)]
        if weight_intializer == 'He':
            He()
        elif weight_intializer == 'Xavier':
            Xavier()
        else:
            raise ValueError(f'Unknown weight initialization : {weight_intializer}')

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, Layers):
        self.x = None
        self.y = None
        self.layers = Layers
        self.params = self.parameters()

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def fit(self,x,y,optimizer = 'adam',epochs = 1000):
        self.x = x
        self.y = y
        self.optimizer = optimizer
        y_pred = [self(xs) for xs in self.x]
        cost = sum((yout - ygt) ** 2 for ygt, yout in zip(self.y, y_pred))
        self.m = [0]*len(self.parameters())
        self.v = [0]*len(self.parameters())
        self.epoch = 1
        self.epoch_limit = epochs

        def adam(learning_rate = 0.01):
            self.m =[0.9*w1 + 0.1*grad for w1,grad in zip(self.m, [p.grad for p in self.parameters()])]
            self.v =[0.999*v1 + 0.001*(grad**2) for v1,grad in zip(self.v, [p.grad for p in self.parameters()])]
            hat_m = [_m / (1 - 0.9 ** self.epoch) for _m in self.m]
            hat_v = [_v / (1 - 0.999 ** self.epoch) for _v in self.v]
            r_v = [np.sqrt(1/(_hat_v + 0.1**4)) for _hat_v in hat_v]
            for params,_hat_m,_r_v in zip(self.parameters(),hat_m,r_v):
                params.data -= learning_rate*_hat_m*_r_v

        def RMSprop(learning_rate = 0.001):
            self.v =[0.999*v1 + 0.001*(grad**2) for v1,grad in zip(self.v, [p.grad for p in self.parameters()])]
            r_v = [(1/(np.sqrt(_hat_v) + 0.1**4)) for _hat_v in self.v]
            for params,_r_v in zip(self.parameters(),r_v):
                params.data -= learning_rate*_r_v*params.grad

        def momentum(learning_rate = 0.01):
            self.m =[0.9*w1 + 0.1*grad for w1,grad in zip(self.m, [p.grad for p in self.parameters()])]
            for params,_hat_m in zip(self.parameters(),self.m):
                params.data -= learning_rate*_hat_m

        if self.optimizer == 'adam':
            update_step = adam
        elif self.optimizer =='RMSprop':
            update_step = RMSprop
        elif self.optimizer =='momentum':
            update_step = momentum
        else:
            raise ValueError(f'Unknown Optimizer : {self.optimizer}')
        loss = []
        lr = 0.005
        patience = 0
        while self.epoch <= self.epoch_limit:
            self.zero_grad()
            y_pred = [self(xs) for xs in self.x]
            cost = sum([(yout - ygt) ** 2 for ygt, yout in zip(self.y, y_pred)])
            cost = cost * (1.0/len(y_pred))
            cost.backward()
            update_step(lr)
            print(f"Epoch {self.epoch} | Loss: {cost.data}")
            loss.append(cost.data)
            self.epoch += 1
            if ((len(loss) > 3 and np.abs((loss[-2] - loss[-1])/loss[-1]  < 0.05) and cost.data < 0.2 )and lr*0.8 >1e-5 ):
              lr = lr*0.8
              print(f"\n teduced lr to {lr}")
            if(len(loss) > 3 and loss[-2] < loss[-1]):
              patience += 1
              if (patience > 3 ) or cost.data < 0.01:
                patience = 0
                print("/n Early stopping")
                break
        return loss
    def predict(self,x):
      preds = []
      for xs in x:
          # convert to Neuron if needed
          xs_neuron = [Neuron(xi) if not isinstance(xi, Neuron) else xi for xi in xs]
          y_pred = self(xs_neuron)  # forward pass
          preds.append(y_pred.data)
      return preds
class BatchNorm1dNode(Module):

    def __init__(self, n_features):

        self.gamma = [Neuron(1.0) for _ in range(n_features)]
        self.beta  = [Neuron(0.0) for _ in range(n_features)]
        self.n_features = n_features

    def __call__(self, x):

        assert len(x) == self.n_features

        mean = sum(x, Neuron(0.0)) / self.n_features
        var_terms = [(xi - mean) ** 2 for xi in x]
        var = sum(var_terms, Neuron(0.0)) / self.n_features
        std = (var + 1e-5) ** 0.5

        out = []
        for xi, g, b in zip(x, self.gamma, self.beta):
            x_hat = (xi - mean) / std
            yi = g * x_hat + b
            out.append(yi)
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return self.gamma + self.beta

    def __repr__(self):
        return f"BatchNorm1dNode({self.n_features})"


class BatchNorm1d(Module):

    def __init__(self, n_features, **kwargs):
        self.nodes = [BatchNorm1dNode(n_features)]

    def __call__(self, x):
        out = [n(x) for n in self.nodes]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.nodes for p in n.parameters()]

    def __repr__(self):
        return f"BatchNorm1d of [{', '.join(str(n) for n in self.nodes)}]"

class LayerNorm1dNode(Module):

    def __init__(self, n_features, eps=1e-5):
        # gamma and beta for each feature (like weights + bias)
        self.gamma = [Neuron(1.0) for _ in range(n_features)]
        self.beta  = [Neuron(0.0) for _ in range(n_features)]
        self.n_features = n_features
        self.eps = eps

    def __call__(self, x):

        assert len(x) == self.n_features


        mean = sum(x, Neuron(0.0)) * (1.0 / self.n_features)


        var_terms = [(xi - mean) ** 2 for xi in x]
        var = sum(var_terms, Neuron(0.0)) * (1.0 / self.n_features)

        std = (var + self.eps) ** 0.5

        out = []
        for xi, g, b in zip(x, self.gamma, self.beta):
            x_hat = (xi - mean) / std
            yi = g * x_hat + b
            out.append(yi)

        return out[0] if len(out) == 1 else out

    def parameters(self):
        return self.gamma + self.beta

    def __repr__(self):
        return f"LayerNorm1dNode({self.n_features})"

class LayerNorm1d(Module):


    def __init__(self, n_features, **kwargs):
        self.nodes = [LayerNorm1dNode(n_features)]

    def __call__(self, x):
        # x: list[Neuron] for ONE sample
        out = [n(x) for n in self.nodes]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.nodes for p in n.parameters()]

    def __repr__(self):
        return f"LayerNorm1d of [{', '.join(str(n) for n in self.nodes)}]"


