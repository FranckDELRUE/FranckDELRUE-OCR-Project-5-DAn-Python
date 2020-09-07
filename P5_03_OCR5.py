import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
from scipy.spatial.distance import pdist


def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()
    
def ligneToColonne(df, liste):
    """
        Prends une data frame et une liste. Permet de mettre des éléments du dictionnaire, présentés en ligne, en colonne.
        Retourne la data Frame avec les colonnes.
        Liste[0] est le Code Produit
        Liste[1] est le Code Élément
        Liste[2] est le nom de la nouvelle colonne
        Liste[3] sont les colonnes à merger
    """
    
    filt = (df['Code Produit'] == liste[0]) & (df['Code Élément'] == liste[1])
    colonne = liste[3]
    avec = ['Zone']

    df = df.merge(df.loc[filt, colonne], how = 'left', on = avec).fillna(0)
    df.rename(columns={liste[3][1] + '_x': liste[3][1], liste[3][1] + '_y': liste[2]}, inplace=True)
    
    return df

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(20,20))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
            
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, data, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(20,20))
        
            color1=[0, 0, 1, 1]
            color2=[1, 0.65, 0, 1]
            color3=[0, 1, 0, 1]
            color4=[1, 0, 0, 1]
            color5=[0.29, 0, 0.51, 1]

            colormap = np.array([color1, color2, color3, color4, color5])
            
            X_projected = np.hstack((X_projected, np.atleast_2d(data).T))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha, c = colormap[data-1])
                meanD1 = 0
                meanD2 = 0
                for i in range(0,int(data.max())):
                    meanD1 = X_projected[X_projected[ : , -1] == i+1][:, d1].mean()
                    meanD2 = X_projected[X_projected[ : , -1] == i+1][:, d2].mean()
                    plt.scatter(meanD1, meanD2, marker = '^', s = 200, alpha=alpha, c = colormap[i])
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='10', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
            
def chi_2(X, Y, df):
    cont = pd.pivot_table(df[[X, Y]], index=X ,columns=Y, aggfunc=len, margins=True, margins_name="Total")

    tx = cont.loc[:,["Total"]]
    ty = cont.loc[["Total"],:]
    n = len(df)
    indep = tx.dot(ty) / n

    c = cont.fillna(0) # On remplace les valeurs nulles par 0
    measure = (c-indep)**2/indep
    xi_n = measure.sum().sum()
    table = measure/xi_n
    
    return (table, c)

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT

def lorenz(df):
    dep = df.values
    n = len(dep)
    lorenz = np.cumsum(np.sort(dep)) / dep.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0
    
    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n # Surface sous la courbe de Lorenz. Le premier segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le dernier segment lorenz[-1] qui est à moitié au dessus de 1.
    S = 0.5 - AUC # surface entre la première bissectrice et le courbe de Lorenz
    gini = 2*S
    
    return (lorenz, n, gini)

def coefCophenetic(df):
    """
        Prends une data frame. Permet de calculer les coefficients Cophenetic de chaque méthode
        pour un dataFrame donné pour la fonction sklearn.linkage.
        Les DataFrames doivent être centré et réduit
        Retourne un DataFrame avec les coefficients
    """
    methods = ['Single', 'Complete', 'Average', 'Weighted', 'Centroid', 'Median', 'Ward']
    copheneticCoef = []
    for method in methods:
        Z = linkage(df, method.lower())
        c, coph_dists = cophenet(Z, pdist(df))
        copheneticCoef.append(round(c,2))
        
    return pd.DataFrame(data=[copheneticCoef], columns=methods, index=['Cophenetic Coefficient'])

def coefGini(df):
    """
        Prends une data frame. Permet de calculer les coefficients Gini de chaque méthode
        pour un dataFrame donné pour la fonction sklearn.linkage.
        Les DataFrames doivent être centré et réduit
        Retourne un DataFrame avec les coefficients
    """
    methods = ['Single', 'Complete', 'Average', 'Weighted', 'Centroid', 'Median', 'Ward']
    giniCoeff = []
    for method in methods:
        Z = linkage(df, method.lower())
        _, _, gini = lorenz(pd.DataFrame(data=Z[:,2]))
        giniCoeff.append(gini)

    return pd.DataFrame(data=[giniCoeff], columns=methods, index=['Gini Coefficient'])

def plotbox(data, clusters):   
    data['Clusters'] = clusters
    data_all = data.copy()
    data_all['Clusters'] = 0
    data = data.append(data_all, ignore_index=True)
    data.sort_values(by=['Clusters'], inplace = True)
    data.loc[data['Clusters'] == 0, 'Clusters'] = 'All'

    for colonne in data.columns:
        if colonne != 'Clusters':

            fig, axes = plt.subplots(figsize=(20, 16))

            mu = data[colonne].mean()
            sigma = data[colonne].std()

            df = (data[colonne] - mu) / sigma

            fig.suptitle('Moyenne de la '+colonne+ ' en fonction du cluster', fontsize= 18)

            ax1 = sns.boxplot(x=data['Clusters'], y=df, showmeans=True)
            ax2 = sns.swarmplot(x=data['Clusters'], y=df, color=".25")
            plt.axvline(0.5)

            plt.xlabel("Cluster N°")
            plt.ylabel(colonne +" (σ)")

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            i = 0

            for cluster in data['Clusters'].unique():

                mu_clusters = data.loc[data['Clusters'] == cluster, colonne].mean()
                sigma_clusters = data.loc[data['Clusters'] == cluster, colonne].std()

                textstr = '\n'.join((
                    r'Population : ',
                    r'$\mu=%.2f$' % (mu_clusters, ),
                    r'$\sigma=%.2f$' % (sigma_clusters, ),
                    r'$n=%.0f$' % (len(data.loc[data['Clusters'] == cluster, colonne]))))

                # place a text box in upper left in axes coords
                axes.text(0.05 + 0.99 / len(data['Clusters'].unique()) * i, 1.09, textstr, transform=axes.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)

                i = i + 1

            plt.show()
            
def NettoyagePays(df):
    # Nous enlevons les données concernant des groupements de pays (Code zone > 5000)
    df = df[~(df['Code zone'] >= 5000)]

    # Nous enlevons les données concernant la Chine qui est un groupement de pays (Code Zone = 351)
    df = df[~(df['Code zone'] == 351)]
    
    return df


def ElementAGarder(df, element, produit):
    df = df[df['Code Produit'].isin(produit)]
    df = df[df['Code Élément'].isin(element)]
    
    return df
