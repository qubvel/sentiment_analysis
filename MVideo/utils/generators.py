import numpy as np

def train_generator(X, y, w2v_converter, samples_per_epoch=None, batch_size=8, shuffle=False, reverse_prob=False):
    while True:
        if shuffle:
            indicies = np.random.permutation(len(X))
        else:
            indicies = list(range(len(X)))


        order = 1   
        if reverse_prob:
            value = np.random.randint(100)
            if value < np.clip((reverse_prob*100), 0, 100):
                oreder = -1

        n_batch = (len(X) + batch_size - 1) // batch_size - 1

        for i in range(n_batch):

            ind = indicies[i*batch_size : (i+1)*batch_size]

            X_yield = X[ind]
            y_yield = y[ind]
            
            X_yield = list(map(lambda x: w2v_converter.convert2matrix(x), X_yield))

            X_yield = np.array(X_yield, dtype=np.float32)
            X_yield[:, ::order]

            y_yield = np.array(y_yield, dtype=np.float32) 
            
            yield (X_yield, y_yield)

def test_generator(X, w2v_converter, samples_per_epoch=None):

    for x in X:
        x = w2v_converter.process(x)
        x = w2v_converter.convert2matrix(x)
        yield x