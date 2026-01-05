class Neuron:
    __slots__ = ['data', 'grad', 'children', '_backward']
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

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

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

    def log(self):
        out = Neuron(np.log(np.maximum(self.data, 1e-12)), (self,))
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Neuron(np.exp(self.data), (self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Neuron({self.data} | grad {self.grad})"


    def backward(self):

        topo = []
        visited = set()

        stack = [(self, 0)]

        while stack:
            node, state = stack.pop()
            if state == 0:
                if node in visited:
                    continue
                visited.add(node)

                stack.append((node, 1))

                for child in node.children:
                    if isinstance(child, Neuron) and child not in visited:
                        stack.append((child, 0))
            else:
                topo.append(node)


        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Node(Module):
    __slots__ = ['w', 'b', 'nin', 'nonlin']
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
class Softmax(Module):
    def __init__(self, nfeatures):
        self.nfeatures = nfeatures

    def __call__(self, x):
        exp_x = [a.exp() for a in x]
        exp_sum = sum(exp_x)
        return [expi / exp_sum for expi in exp_x]

def sparse_categorical_crossentropy(logits, target):
    max_val = max(l.data for l in logits)
    shifted_logits = [l - max_val for l in logits]
    logitsexp = [logit.exp() for logit in logits]
    total = sum(logitsexp)
    probs = [le / total for le in logitsexp]
    return probs[int(target)].log() * Neuron(-1.0)


class MLP(Module):

    def __init__(self, Layers):
        self.x = None
        self.y = None
        self.layers = Layers
        self.params = self.parameters()
        self.m = [0]*len(self.params)
        self.v = [0]*len(self.params)
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def fit(self,x,y,optimizer = 'adam',epochs = 1000,batchsize = 256,lrate = 0.01):
        self.x = x
        self.y = y
        self.optimizer = optimizer
        self.epoch = 1
        self.epoch_limit = epochs

        def gd(learning_rate = 0.001):
          for param in self.parameters():
            param.data -= learning_rate*param.grad

        def adam(learning_rate = 0.001):
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
        elif self.optimizer == 'gd':
          update_step = gd
        else:
            raise ValueError(f'Unknown Optimizer : {self.optimizer}')
        loss = []
        lr = lrate
        patience = 0
        i = 0
        batch_size = batchsize
        while self.epoch <= self.epoch_limit:
          epoch_loss = Neuron(0.0)
          for _ in range(0,self.x.shape[0],batch_size):
            batch = self.x[_:min(_+batch_size,self.x.shape[0] - 1)]
            self.zero_grad()
            y_pred = np.array([self(x) for x in batch])
            scce = []
            for logits,c_target in zip(y_pred,self.y[_:min(_+batch_size,self.x.shape[0] - 1)]):
                scce.append(sparse_categorical_crossentropy(logits,c_target))
            cost_t = sum(scce)
            cost = cost_t * (1.0/len(y_pred))
            epoch_loss += cost
            cost.backward()
            print(f"Batch {(_//batch_size + 1)} | Loss: {cost.data}    ",end = '\r')
            update_step(lr)
            for p in self.parameters():
              p.children = () # Break references
              p._backward = lambda: None
          print(f"Epoch {self.epoch} | Loss: {epoch_loss.data/(self.x.shape[0]//batch_size)}")
          loss.append(cost.data)
          cost = None
          y_pred = None
          scce = None
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
          preds.append(y_pred.data if isinstance(y_pred,Neuron) else y_pred)
      return np.array(preds)
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


