"""
Visualization functions for volatility surface and latent space analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_vol_surface(df):
    sns.heatmap(df.pivot(index='strike', columns='time_to_expiry', values='vol'), cmap='viridis')
    plt.title('Volatility Surface')
    plt.show()

def plot_latent_space(z_mean):
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_mean)
    plt.scatter(z_pca[:, 0], z_pca[:, 1])
    plt.title('Latent Space (PCA)')
    plt.show()
