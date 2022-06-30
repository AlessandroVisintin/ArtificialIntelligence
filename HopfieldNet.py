import numpy as np


class HopfieldNet:
    
    # Hopfield (1982)
    # Simple implementation of an Hopfield Network 
    
    def __init__(self, neurons=5):
        self.neurons = np.ones(neurons, dtype="int8") # {-1,1} states
        self.T = np.zeros((neurons, neurons)) # weights matrix
        self.U = np.zeros(neurons) # neurons thresholds
        self.I = np.identity(neurons) # identity matrix
    
    
    #set neuron value at given index
    # value must be {-1,1}
    def set_neuron(self, index, value):
        if value not in [1,-1]:
            raise RuntimeError(f'Invalid value for a neuron: {value}.')
        self.neurons[index] = value
    
    
    # set weights matrix from input
    # matrix must be symmetric
    def set_weights(self, matrix):
        matrix = np.array(matrix)
        if matrix.shape != self.T.shape:
            raise RuntimeError('Incompatible shape for weights matrix.')
        if not np.all(np.abs(matrix - matrix.T) < 1e-8):
            raise RuntimeError('Weights matrix is not symmetric.')
        self.T = matrix


    # stores binary patterns inside the network
    # overwrites previous matrix
    def store_sequences(self, sequences):
        tmp_matrix = np.zeros(self.T.shape)
        for s in sequences:
            s = np.array([s])
            tmp_matrix += np.outer(s, s) - self.I
        self.T = tmp_matrix / len(sequences)

    
    # update neurons states until stable point is reached
    # negative weights -> neurons diverge to opposite values
    # positive weights -> neurons converge to same values
    def update_state(self, verbose=False):
        cycle, stable = 1, False
        while not stable:
            print(f'update cycle: {cycle}') if verbose else None
            cycle, stable = cycle + 1, True
            for i in range(self.U.shape[0]):
                if np.dot(self.T[i,:], self.neurons) >= self.U[i]:
                    if self.neurons[i] < 0: 
                        self.neurons[i] = 1
                        stable = False
                else:
                    if self.neurons[i] > 0: 
                        self.neurons[i] = -1
                        stable = False

    # return current state of neurons
    def instantaneous_state(self):
        return self.neurons
    
    
    # global energy of the net
    def energy(self):
        inter = 0.5 * self.neurons[:,None] * self.T * self.neurons
        intra = self.U[:,None] * self.neurons 
        return - inter.sum() - intra.sum()

    
    # test function to evaluate error rate
    # paper suggests no more than 0.15*nodes patterns
    def test(self, store_seq=20, tot_seq=2**20):
        n = self.neurons.shape[0]
        store = n
        tot = 2**n
        store = store_seq if store > store_seq else store
        tot = tot_seq if tot > tot_seq else tot
        
        seq = [[-1 if x == '0' else 1 for x in f'{i:0{n}b}'] for i in range(tot)]
        seq = np.array(seq)
        pat = [tuple(x) for x in seq[np.random.randint(0, tot, size=store)]]
        self.store_sequences(pat)

        E = []
        for s in seq:
            for idx, val in enumerate(s):
                self.set_neuron(idx, val)
            E.append(self.energy())
        
        seq = [(x,y) for x,y in zip(seq, E)]
        seq = sorted(seq, key=lambda t:t[1])
        
        res = []
        for i,(s,e) in enumerate(seq):
            if tuple(s) in pat:
                res.append(i)
        
        print(f'Higher energy pattern at position {max(res)} out of {tot}')
        

if __name__ == '__main__':
    
    hnet = HopfieldNet(10)
    hnet.test(store_seq=2) 

