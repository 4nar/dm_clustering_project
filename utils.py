import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from sklearn import preprocessing
from sklearn.cluster import KMeans


def distance(arr, oth_arr, type='euclidean'):
    k = oth_arr.shape[0]
    n = arr.shape[0]
    diff_matrix = np.zeros((k, n))
    if type=='euclidean':
        for i in range(k):
            diff = (arr - oth_arr[i])
            diff_prod = diff.dot(diff.T)
            diff_matrix[i] = np.sqrt(diff_prod[np.diag_indices(n)])
    if type=='manhattan':
        for i in range(k):
            diff = abs(arr - oth_arr[i])
            diff_matrix[i] = np.sum(diff, axis=1)
            
    return diff_matrix.T


def kmeans(X, k):
    n = X.shape[0]
    centroids = X[np.random.randint(0, n, k), :]
    num_iter = 300
    labels = np.argmin(distance(X, centroids), axis=1)
    for iter in range(num_iter):
        for i in range(k):
            centroids[i] = X[np.where(labels==i)[0],:].mean(axis=0)
        new_labels = np.argmin(distance(X, centroids), axis=1)
        if np.array_equal(labels, new_labels):
            return labels
        labels = new_labels
    return labels

def distance(arr, type='euclidean'):
    n = arr.shape[0]
    diff_matrix = np.zeros((n,n))
    if type=='euclidean':
        for i in range(n):
            diff = (arr - arr[i])
            diff_prod = diff.dot(diff.T)
            diff_matrix[i] = np.sqrt(diff_prod[np.diag_indices(n)])
    return diff_matrix
def spectral_clustering_laplace_norm(X, sigma=0.3, k=3):
    pairwise_differences = distance(X)
    # Constructing similarity graph 
    similarity_graph = np.exp(-(pairwise_differences)/(2*sigma**2))
    # Computing normalized Laplacian
    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=True)
    L_norm_cov = np.cov(graph_laplacian_s)
    eig_val, eig_vectors = eigh(L_norm_cov)
    X = eig_vectors[:, :k]
    X_normalized =  preprocessing.normalize(X, norm='l2')
    model = KMeans(n_clusters=k)
    labels_norm = model.fit_predict(X_normalized)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(df['A'], df['B'], c=labels_norm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Figure 1: Relation between Variables, labeled using spectral clustering with sigma={sigma}')
    return labels_norm