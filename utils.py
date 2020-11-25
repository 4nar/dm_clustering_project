import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pandas as pd

def nmi(y, labels):
    a = np.vstack((labels, y)).T
    df_mut = pd.DataFrame(a, columns=['C', 'D'])
    df_mut['P(C)']=df_mut['C'].map(df_mut['C'].value_counts(normalize=True))
    df_mut['P(D)']=df_mut['D'].map(df_mut['D'].value_counts(normalize=True))
    joint_prob = df_mut.groupby(['C','D'], as_index=False).size()/df_mut.shape[0]
    df_mut = df_mut.set_index(['C','D']).join(joint_prob.to_frame())
    df_mut.rename(columns={0:'P(C,D)'}, inplace=True)
    df_mut = df_mut.drop_duplicates()
    nom = np.sum(df_mut['P(C,D)']*np.log2(df_mut['P(C,D)']/(df_mut['P(C)']*df_mut['P(D)'])))
    df_mut['H(C)'] = df_mut['P(C)'] * np.log2(df_mut['P(C)'])
    df_mut['H(D)'] = df_mut['P(D)'] * np.log2(df_mut['P(D)'])
    denom = np.sqrt(df_mut['H(C)'].sum()*df_mut['H(D)'].sum())
    nmi = nom/denom
    return nmi

def distance_pair(arr, oth_arr, type='euclidean'):
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


def kmeans(X, k, random_state=123):
    np.random.seed(random_state)
    n = X.shape[0]
    centroids = X[np.random.randint(0, n, k), :]
    num_iter = 300
    labels = np.argmin(distance_pair(X, centroids), axis=1)
    for iter in range(num_iter):
        for i in range(k):
            centroids[i] = X[np.where(labels==i)[0],:].mean(axis=0)
        new_labels = np.argmin(distance_pair(X, centroids), axis=1)
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
    
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax.scatter(df['A'], df['B'], c=labels_norm)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_title(f'Figure 1: Relation between Variables, labeled using spectral clustering with sigma={sigma}')
    return labels_norm