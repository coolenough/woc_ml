import numpy as np
class PolynomialRegression:
    def transform(self,x):
        b = None
        for i in range(self.degree):
            b = np.stack((x,x**i) , axis = 1)
            a = b.shape
            b = np.reshape(b, newshape=(a[0],a[1]*a[2]))
        x = b
        return x
    def __init__(self,optimizer = "RMSprop",degree = 3):
        self.error = None
        self._y = None
        self.b = None
        self.w = None
        self.y = None
        self.x = None
        self.w_error = None
        self.b_error = None
        self.m_w = None
        self.m_b = None
        self.v_w = None
        self.v_b = None
        self.r_v_w = None
        self.r_v_b = None
        self.hat_m_w = None
        self.hat_m_b = None
        self.epoch_count = 1
        self.epoch_limit = None
        self.hat_v_w = None
        self.hat_v_b = None
        self.update_step = lambda: None
        self.metrics = self.Metrics(self)
        self.optimiser = optimizer
        self.degree = degree
    def fit(self,xs,ys,epochs = 100,):
        self.x = xs.copy()
        self.y = ys.copy()
        self.x = self.transform(self.x)
        self.w = np.zeros(self.x.shape[1])
        self.b = 0.0
        self.v_w = np.zeros(self.x.shape[1])
        self.m_w = np.zeros(self.x.shape[1])
        self.m_b = 0
        self.v_b = 0
        self._y = np.dot(self.x , self.w) + self.b
        self.error = np.mean((self._y - self.y)**2)
        self.b_error = 2*np.mean(self._y - self.y)
        self.w_error = 2*np.dot(self.x.transpose(),self._y - self.y)/self.x.shape[0]
        self.epoch_limit = epochs
        if self.optimiser == "adam":
            update_step = self.adam
        elif self.optimiser == "RMSprop":
            update_step = self.RMSprop
        elif self.optimiser == "momentum":
            update_step = self.momentum
        elif self.optimiser == "gd":
            update_step = self.gd
        else:
            raise ValueError(f"Unknown optimizer: {self.optimiser}")
        while self.epoch_count <= self.epoch_limit:
            update_step(gamma = 0.0005)
            self._y = np.dot(self.x,self.w) + self.b
            self.error = np.mean((self._y - self.y)**2)
            self.b_error = 2*np.mean(self._y - self.y)
            self.w_error = 2*np.dot(self.x.transpose(),self._y - self.y)/self.x.shape[0]
            self.epoch_count += 1
            print(f"Epochs {self.epoch_count -1} / {self.epoch_limit} | loss : {self.error}")
        self.metrics = self.Metrics(self)
    def adam(self, beta1 = 0.9 , beta2 = 0.999 ,epsilon = 10**-8,gamma = 0.001):
        #momentum variables
        self.m_w = beta1*self.m_w + (1 - beta1)*self.w_error
        self.m_b = beta1*self.m_b + (1 - beta1)*self.b_error
        #RMSpropFactors
        self.v_w = beta2*self.v_w + (1-beta2)*(self.w_error**2)
        self.v_b = beta2*self.v_b + (1-beta2)*(self.b_error**2)
        #bias correcting the variance values
        self.hat_v_w = self.v_w / (1 - beta2**self.epoch_count)
        self.hat_v_b = self.v_b / (1 - beta2**self.epoch_count)
        #RMSpropFactvectors for updating the final values
        self.r_v_w = 1 / (np.sqrt(self.hat_v_w + epsilon))
        self.r_v_b = 1 / (np.sqrt(self.hat_v_b + epsilon))
        #bias correcting the momentum values
        self.hat_m_w = self.m_w / (1 - beta1**self.epoch_count)
        self.hat_m_b = self.m_b / (1 - beta1**self.epoch_count)
        #Updating the final value
        self.b = self.b - self.hat_m_b*self.r_v_b*gamma
        self.w = self.w - np.multiply(self.hat_m_w,self.r_v_w)*gamma
    def RMSprop(self,beta2 = 0.999,epsilon = 10**-8, gamma = 0.001):
        self.v_w = beta2*self.v_w + (1-beta2)*(self.w_error**2)
        self.v_b = beta2*self.v_b + (1-beta2)*(self.b_error**2)
        self.r_v_w = 1 / (np.sqrt(self.v_w + epsilon))
        self.r_v_b = 1 / (np.sqrt(self.v_b + epsilon))
        self.b = self.b - self.b_error*self.r_v_b*gamma
        self.w = self.w - np.multiply(self.w_error,self.r_v_w)*gamma
    def momentum(self,beta1 = 0.9 , gamma = 0.001):
        self.m_w = beta1*self.m_w + (1 - beta1)*self.w_error
        self.m_b = beta1*self.m_b + (1 - beta1)*self.b_error
        self.b = self.b - self.m_b*gamma
        self.w = self.w - self.m_w*gamma
    def gd(self,gamma = 0.001):
        self.w = self.w - gamma*self.w_error
        self.b = self.b - self.b_error*gamma
    def predict(self,x):
        x = self.transform(x)
        out = np.dot(x , self.w) + self.b
        return out
    class Metrics:
        def __init__(self,outer):
            self.parameters = None
            self.outer = outer
            self.w = self.outer.w
            self.b = self.outer.b
        def params(self):
            self.parameters = np.append(self.w , [self.b], axis = None)
            return self.parameters
        def r2_score(self, x, y):
            y_pred = self.outer.predict(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        def corr(self,x,y):
            y_pred = self.outer.predict(x)
            return np.corrcoef(y_pred,y)