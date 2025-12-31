import numpy as np
class LogisticRegression:
    def __init__(self):
        self.optimizer = None
        self.hat_m_b = None
        self.hat_m_w = None
        self.r_v_b = None
        self.r_v_w = None
        self.hat_v_w = None
        self.hat_v_b = None
        self.x = None
        self.y = None
        self._y = None
        self.w = None
        self.b = None
        self.w_error = None
        self.b_error = None
        self.m_w = None
        self.m_b = None
        self.v_w = None
        self.v_b = None
        self.epoch_count = 1
        self.epoch_limit = None
    def predict(self,x):
        out = 1/(1 + np.exp(-(np.dot(x , self.w) + self.b)))
        return out
    def fit(self,x,y,epochs = 100,optimizer = 'gd'):
        self.x = x
        self.y = y
        self.w = np.zeros(x.shape[1])
        self.b = 0
        self._y = self.predict(self.x)
        self.epoch_limit = epochs
        self.optimizer = optimizer
        #initialized the momentum to avoid type error in first iteration
        self.v_w = np.zeros(self.x.shape[1])
        self.m_w = np.zeros(self.x.shape[1])
        self.m_b = 0
        self.v_b = 0
        #####
        self.b_error = -(np.mean(self.y - self._y))
        self.w_error = -(np.dot(self.x.transpose(),self.y - self._y))/self.x.shape[0]
        if self.optimizer == "adam":
            update_step = self.adam
        elif self.optimizer == "RMSprop":
            update_step = self.RMSprop
        elif self.optimizer == "momentum":
            update_step = self.momentum
        elif self.optimizer == "gd":
            update_step = self.gd
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        while self.epoch_count <= self.epoch_limit:
            self._y = self.predict(self.x)
            self.loss = sum([-t_hat*np.log(t) - (1-t_hat)*np.log(1-t) for t , t_hat in zip(self._y,self.y)])/len(self.y)
            self.b_error = -(np.mean(self.y - self._y))
            self.w_error = -(np.dot(self.x.transpose(),self.y - self._y))/self.x.shape[0]
            update_step()
            print(f'Epoch : {self.epoch_count} Loss : {self.loss}')
            self.epoch_count += 1
    def adam(self, beta1 = 0.9 , beta2 = 0.999 ,epsilon = 10**-8,gamma = 0.0003):
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
    def gd(self,gamma = 0.005):
        self.w = self.w - gamma*self.w_error
        self.b = self.b - self.b_error*gamma

