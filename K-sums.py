import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def plot_cluster(X,labels,
                 title=None, 
                 x_label=None,
                 y_label=None,
                 ):

    plt.figure(figsize=(10, 8))

    plt.scatter(X[:,0], X[:,1],
                c = labels,
                )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


######################  start K-sums


def random_init_labels(X, k):
    '''
    return `labels` with values of 0 ~ k-1
    '''
    return np.random.randint(0,k,size=(X.shape[-1]),dtype=int)

def cal_Ds_ns(X,k,labels):
    '''
    calculate D_1, D_2, ..., D_k, & n_1, n_2, ..., n_k
    '''
    Ds = [
        np.sum(X[:,labels==i], axis=1) for i in range(k)
    ]

    ns = [
        X[:,labels==i].shape[-1] for i in range(k)
    ]
    return Ds, ns

def cal_dCw(x_i,w,D,n):
    '''
    calculate d(x_i, C_w)
    '''
    D_w = D[w] 
    n_w = n[w] 
    return (x_i - D_w / n_w) @ (x_i - D_w / n_w) 

def cal_dCv(x_i,v,D,n):
    '''
    calculate d(x_i, C_v)
    '''
    D_v = D[v] 
    n_v = n[v] 
    return (n_v * x_i - D_v) @ (n_v * x_i - D_v) / (n_v+1) ** 2

def find_min_dCv(x_i,w,k,D,n):
    '''
    find the min d_Cv & v,
    * inital assign min_dCv = d_Cw, v = w
    '''
    v_s = np.arange(k).tolist()
    v_s.remove(w) 
    min_dCv = cal_dCw(x_i,w,D,n)
    min_v = w 
    for v in v_s:
        dCv = cal_dCv(x_i,v,D,n)
        if dCv < min_dCv:
            min_dCv = dCv 
            min_v = v 
    return min_dCv,min_v 

def k_sums(X, k=3, max_iter=500):
    '''
    X : column vertors 
    k : numbers of clustering result
    max_iter : max loops for iteration
    '''

    ## init labels
    labels = np.full(len(X[0]), 0)
    labels = random_init_labels(X,k)
    D,n = cal_Ds_ns(X,k,labels)

    ## show labels that just were generated
    plot_cluster(X.T, labels,
                    title=f"before Iteration",
                    x_label="X1",
                    y_label="X2",
                    )
    plt.show()

    ## some helpers 
    iter = 1
    lables_changed = True
    pre_labels = labels.copy()

    ## loop for iteration
    while lables_changed and iter < max_iter:
        ## paper says x_i in random order, so we use shuffle
        random_index = np.arange(len(labels))
        np.random.shuffle(random_index)

        for idx in random_index:
            w = labels[idx]
            x_i = X[:,idx]  
            dCw = cal_dCw(x_i,w,D,n)
            dCv,v = find_min_dCv(x_i,w,k,D,n)
            if dCw - dCv > 0:
                labels[idx] = v 
                D[w] -= x_i 
                n[w] -= 1
                D[v] += x_i 
                n[v] += 1
        
        ## if labels don't change, we can finish the code in advance
        if np.allclose(pre_labels,labels):
            lables_changed = False 
        
        ## show clustering process
        plot_cluster(X.T, labels,
                        title=f"Iteration {iter}",
                        x_label="X1",
                        y_label="X2",
                        )
        plt.show()

        ## change helpers 
        pre_labels = labels
        iter += 1

    ## return S_1, S_2, ..., S_k
    return [
        X[:,labels==i] for i in range(k)
    ]


if __name__ == "__main__":
    ## prepare
    real_centroids = [(-4, -4), (0, 0), (4, 4)]
    X, y = make_blobs(n_samples=300, centers=real_centroids, n_features=2, 
                    random_state=42)

    ## K-sums
    X_column = X.T ## X is row vertors, transpose it into column vectors
    S_s = k_sums(X=X_column)
