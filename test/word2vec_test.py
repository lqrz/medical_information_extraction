__author__ = 'root'

import gensim
import logging

from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

W2V_PRETRAINED_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'

def plot_pca(samples, dimensions, color):
    if dimensions == '2D':
        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        plt.plot(sklearn_transf[:,0],sklearn_transf[:,1],\
             'o', markersize=7, color=color, alpha=0.5, label='')
        # plt.plot(sklearn_transf[1::2,0], sklearn_transf[1::2,1],\
        #      '^', markersize=7, color='red', alpha=0.5, label='Matrix')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    #     plt.xlim([-4,4])
        plt.ylim([-.8,.8])
        plt.legend()
        plt.title('Word embeddings PCA')

        print sklearn_transf

    elif dimensions == '3D':
        sklearn_pca = sklearnPCA(n_components=3)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10
        ax.plot(sklearn_transf[:,0], sklearn_transf[:,1],\
            sklearn_transf[:,2], 'o', markersize=8, color='blue', alpha=0.5, label='')
        # ax.plot(sklearn_transf[:,0], sklearn_transf[:,1],\
        #     sklearn_transf[:,2], '^', markersize=8, alpha=0.5, color='red', label='Matrix')

        plt.title('Word embeddings PCA')
        ax.legend(loc='upper right')

        print sklearn_transf

    # plt.savefig("%s-%s.png" % (word, dimensions), bbox_inches='tight', dpi=200)

    # plt.show(dpi=200)

    # plt.close()

    return True

def load_w2v(model_filename):
    return gensim.models.Word2Vec.load_word2vec_format(model_filename, binary=True)

def construct_samples(w2v_model, words):
    x = np.empty((0,w2v_model.layer1_size))

    for word in words:
        try:
            rep = w2v_model[word]
            x = np.r_[x, rep[np.newaxis,:]]
            print x.shape
        except:
            logger.info('No representation for word: '+word)

    return x

if __name__ == '__main__':

    model = load_w2v(W2V_PRETRAINED_FILENAME)

    print 'Myocardial'
    print model.most_similar(positive='myocardial', topn=20)

    print 'Adderall'
    print model.most_similar(positive='Adderall', topn=20)

    print 'Alzheimer'
    print model.most_similar(positive='Alzheimer', topn=20)

    words = ['myocardial', 'Fibrillation', 'Auricular']
    samples = construct_samples(model, words)

    plot_pca(samples, dimensions='2D', color='red')

    words = ['Abilify', 'Adderall', 'Ativan', 'aspirin', 'citalopram']
    samples = construct_samples(model, words)
    plot_pca(samples, dimensions='2D', color='blue')

    plt.show()

    logger.info('End')