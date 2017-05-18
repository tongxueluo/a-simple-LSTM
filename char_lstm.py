__author__ = 'luoyuan'

import numpy as np

def sigmoid(x):
    return 1./(1.+np.exp(-x))
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
def diff_softmax(p):  #assuming i==j
    return p - 1
def diff_tanh(tanhx): #note the input is tanhx
    return 1 - tanhx*tanhx
def diff_sigmoid(sigmoidx): #note the input is sigmoidx
    return sigmoidx*(1-sigmoidx)

class char_lstm:
    def __init__(self, hidden_size = 100, seq_length = 25, learning_rate = 1e-1):
        #hyperparameters
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
    def __model_parameters_initialization(self):
        self.W_hx_f = np.random.randn(self.hidden_size, self.hidden_size+self.vocab_size)*0.01 #hidden+input to forget-gate vector
        self.bf = np.bf = np.zeros((self.hidden_size,1))
        self.W_hx_i = np.random.randn(self.hidden_size, self.hidden_size+self.vocab_size)*0.01 #hidden+input to input-gate vector
        self.bi = np.bf = np.zeros((self.hidden_size,1))
        self.W_hx_cbar = np.random.randn(self.hidden_size, self.hidden_size+self.vocab_size)*0.01 #hidden+input to cell
        self.bcbar = np.bf = np.zeros((self.hidden_size,1))
        self.W_hx_o = np.random.randn(self.hidden_size, self.hidden_size+self.vocab_size)*0.01 #hidden+input to output-gate vector
        self.bo = np.bf = np.zeros((self.hidden_size,1))

        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01  #hidden to output
        self.by = np.bf = np.zeros((self.vocab_size,1))
    def __forward_propagate(self, inputs, hidden_prev, cell_prev):
        self.x_s, self.h_s, self.f_s, self.i_s, self.cbar_s, self.c_s, self.o_s, self.y_s = {},{},{},{},{},{},{},{}
        self.p_s = {}
        self.h_s[-1] = np.copy(hidden_prev)
        self.c_s[-1] = np.copy(cell_prev)

        for t in xrange(self.seq_length):
            self.x_s[t] = np.zeros((self.vocab_size,1))
            self.x_s[t][inputs[t]]=1
            self.f_s[t] = sigmoid(np.dot(self.W_hx_f, np.concatenate([self.h_s[t-1], self.x_s[t]]))+self.bf)
            self.i_s[t] = sigmoid(np.dot(self.W_hx_i, np.concatenate([self.h_s[t-1], self.x_s[t]]))+self.bi)
            self.cbar_s[t] = np.tanh(np.dot(self.W_hx_cbar, np.concatenate([self.h_s[t-1], self.x_s[t]]))+self.bcbar)
            self.c_s[t] = self.f_s[t]*self.c_s[t-1]+self.i_s[t]*self.cbar_s[t]
            self.o_s[t] = sigmoid(np.dot(self.W_hx_o, np.concatenate([self.h_s[t-1], self.x_s[t]]))+self.bo)
            self.h_s[t] = self.o_s[t]*np.tanh(self.c_s[t])
            self.y_s[t] = np.dot(self.Why, self.h_s[t])+self.by
            self.p_s[t] = softmax(self.y_s[t])
   #         print 'check ps', np.sum(self.p_s[t])
    def __cross_entropy(self, proba, target):
        return -np.log(proba[target, 0])
    def __backpropagate(self, targets):

        loss = sum([self.__cross_entropy(self.p_s[t],targets[t]) for t in range(self.seq_length)])
        diff_Whxf, diff_Whxi, diff_Whxcbar, diff_Whxo, diff_Why = np.zeros_like(self.W_hx_f), np.zeros_like(self.W_hx_i),\
                                                               np.zeros_like(self.W_hx_cbar), np.zeros_like(self.W_hx_o), \
                                                               np.zeros_like(self.Why)
        diff_bf, diff_bi, diff_bcbar, diff_bo, diff_by = np.zeros_like(self.bf), np.zeros_like(self.bi), \
                                                      np.zeros_like(self.bcbar), np.zeros_like(self.bo), \
                                                      np.zeros_like(self.by)
        diff_cnext = np.zeros_like(self.c_s)
        diff_hnext = np.zeros_like(self.h_s)
        for t in reversed(xrange(self.seq_length)):  #note the reversed order
            dy = np.copy(self.p_s[t])
            dy[targets[t]] = diff_softmax(dy[targets[t]])         #derivative of the cross entropy for softmax probability
            diff_Why += np.dot(dy, self.h_s[t].T)
            diff_by += dy
            diff_h = np.dot(self.Why.T, dy) + diff_hnext
            diff_o = (diff_h*np.tanh(self.c_s[t])).astype(np.float64)
            diff_c = self.o_s[t]*diff_tanh(np.tanh(self.c_s[t])) + diff_cnext
            diff_f = (diff_c*self.c_s[t-1]).astype(np.float64)
            diff_i = diff_c*self.cbar_s[t]
            diff_cbar = diff_c*self.i_s[t]
            diff_Whxf += np.dot(diff_f*diff_sigmoid(self.f_s[t]).astype(diff_Whxf.dtype),
                                np.concatenate([self.h_s[t-1], self.x_s[t]]).T)
            diff_bf += (diff_f*diff_sigmoid(self.f_s[t])).astype(diff_bf.dtype)
            diff_Whxi += np.dot(diff_i*diff_sigmoid(self.i_s[t]),
                                np.concatenate([self.h_s[t-1], self.x_s[t]]).T).astype(diff_Whxi.dtype)
            diff_bi += (diff_i*diff_sigmoid(self.i_s[t])).astype(diff_bi.dtype)
            diff_Whxcbar += np.dot(diff_cbar*diff_tanh(self.cbar_s[t]),
                                   np.concatenate([self.h_s[t-1], self.x_s[t]]).T).astype(diff_Whxcbar.dtype)
            diff_bcbar += (diff_cbar*diff_tanh(self.cbar_s[t])).astype(diff_bcbar.dtype)

            diff_Whxo += np.dot(diff_o*diff_sigmoid(self.o_s[t]).astype(diff_Whxo.dtype),
                                np.concatenate([self.h_s[t-1], self.x_s[t]]).T)
            diff_bo += (diff_o*diff_sigmoid(self.o_s[t])).astype(diff_bo.dtype)
            diff_cnext = self.f_s[t]*diff_c
            diff_hnext = np.dot(diff_Whxf[:, 0:len(self.h_s[t])], diff_f*sigmoid(self.f_s[t])) + \
                         np.dot(diff_Whxi[:, 0:len(self.h_s[t])], diff_i*sigmoid(self.i_s[t])) + \
                         np.dot(diff_Whxo[:, 0:len(self.h_s[t])], diff_o*sigmoid(self.i_s[t])) + \
                         np.dot(diff_Whxcbar[:, 0:len(self.h_s[t])], diff_cbar*sigmoid(self.cbar_s[t]))
            np.clip(diff_hnext, -5, 5, out=diff_hnext)    # prevent exploding

        for dparam in [diff_Whxf, diff_Whxi, diff_Whxcbar, diff_Whxo, diff_Why, diff_bf, diff_bi, diff_bo, diff_bcbar, diff_by]:
            np.clip(dparam, -5, 5, out=dparam) #clip to mitigate exploding gradients
        return loss, diff_Whxf, diff_Whxi, diff_Whxcbar, diff_Whxo, diff_Why, \
               diff_bf, diff_bi, diff_bcbar, diff_bo, diff_by, self.h_s[self.seq_length-1], self.c_s[self.seq_length-1]

    def __adagrad(self, zip_params):
        epsilon = 1e-8
        i = 0
        for param, dparam, mem in zip_params:
            mem += dparam*dparam
            param += -self.learning_rate*dparam/np.sqrt(mem+epsilon)

    def __get_sample(self,h,c,seed_index, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_index] = 1
        indexes = []
        for t in xrange(n):
            f = sigmoid(np.dot(self.W_hx_f, np.concatenate([h, x]))+self.bf)
            i = sigmoid(np.dot(self.W_hx_i, np.concatenate([h,x]))+self.bi)
            cbar = np.tanh(np.dot(self.W_hx_cbar, np.concatenate([h,x]))+self.bcbar)
            c = f*c+i*cbar
            o = sigmoid(np.dot(self.W_hx_o, np.concatenate([h,x]))+self.bo)
            h = o*np.tanh(c)
            y = np.dot(self.Why, h)+self.by
            p = softmax(y)
            index = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size,1))
            x[index] = 1
            indexes.append(index)
        return indexes


    def train(self, train_input_path):
        data = open(train_input_path,'r').read()
        chars = list(set(data))
        data_size, self.vocab_size = len(data), len(chars)
        print 'data has %d characters, %d unique ones'%(data_size,self.vocab_size)
        char_to_index = {ch:i for i,ch in enumerate(chars)}
        index_to_char = {i:ch for i,ch in enumerate(chars)}
        self.__model_parameters_initialization()

        self.n, p = 0, 0
        m_Whxf, m_Whxi, m_Whxcbar, m_Whxo, m_Why = np.zeros_like(self.W_hx_f), np.zeros_like(self.W_hx_i), \
                                                 np.zeros_like(self.W_hx_cbar), np.zeros_like(self.W_hx_o), np.zeros_like(self.Why)
        m_bf, m_bi, m_bcbar, m_bo, m_by = np.zeros_like(self.bf), np.zeros_like(self.bi), \
                                       np.zeros_like(self.bcbar), np.zeros_like(self.bo), np.zeros_like(self.by) # memory variables for Adagrad
     #   smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length  #loss at iteration 0
        while True:
            #prepare inputs (sweeping from left to right in steps seq_length long)
            if p+self.seq_length+1>=len(data) or self.n==0:
                hidden_prev = np.zeros((self.hidden_size, 1))
                cell_prev = np.zeros((self.hidden_size, 1))   #reset LSTM memory
                if self.n!=0: p = p+self.seq_length+1-len(data)
                else: p = 0
            inputs = [char_to_index[ch] for ch in data[p:p+self.seq_length]]
            if len(inputs)<self.seq_length:
                self.seq_length = len(inputs)-1
            targets = [char_to_index[ch] for ch in data[p+1:p+self.seq_length+1]]
            self.__forward_propagate(inputs=inputs, hidden_prev=hidden_prev, cell_prev=cell_prev)
            loss, diff_Whxf, diff_Whxi, diff_Whxcbar, diff_Whxo, diff_Why, \
            diff_bf, diff_bi, diff_bcbar, diff_bo, diff_by, hidden_prev, cell_prev \
                = self.__backpropagate(targets)
         #   smooth_loss = smooth_loss*0.999 + loss*0.001   #just looks nicer
            if self.n%1000 == 0:
                print 'iter %d, loss: %f' % (self.n,  loss) # print progress

            zip_params =  zip([self.W_hx_f, self.W_hx_i, self.W_hx_cbar, self.W_hx_o, self.Why,
                               self.bf, self.bi, self.bcbar, self.bo, self.by],
                              [diff_Whxf, diff_Whxi, diff_Whxcbar, diff_Whxo, diff_Why,
                               diff_bf, diff_bi, diff_bcbar, diff_bo, diff_by],
                              [m_Whxf, m_Whxi, m_Whxcbar, m_Whxo, m_Why, m_bf, m_bi, m_bcbar, m_bo, m_by])

            self.__adagrad(zip_params)

            p += self.seq_length
            self.n += 1

            if self.n%1000 == 0:
                sample_index = self.__get_sample(h=hidden_prev,c=cell_prev, seed_index=inputs[0], n=100)
                txt = ''.join(index_to_char[ix] for ix in sample_index)
                print '------\n %s \n--------'%(txt, )




if __name__ == '__main__':
    charlstm = char_lstm()
    train_input_path = "./light_test.txt"
 #   train_input_path = "./shakespear.txt"
 #   train_input_path = './sonnet18.txt'
    charlstm.train(train_input_path)


