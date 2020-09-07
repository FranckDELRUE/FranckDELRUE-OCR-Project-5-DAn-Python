import pandas as pd
import P5_03_OCR5 as OCR

# ---------------------- Fichier des 4 premières variables ----------------------------------------------

# Nous importons le fichiers comprennat les 4 variables de base
df_data = OCR.NettoyagePays(pd.read_csv('./Data/data_raw.csv',encoding = "ISO-8859-1"))

# Nous gardons que les lignes qui nous intéressent
codeElementAGarder = [511, 664, 674, 5511, 5611, 5911]
codeProduitAGarder = [2501, 2901, 2941, 2734]

df_data = OCR.ElementAGarder(df_data, codeElementAGarder, codeProduitAGarder)

# Nous calculons la différence de populations en %
df_data['Difference de population'] = (df_data['Y2017'] - df_data['Y2014']) / df_data['Y2017'] * 100

# Nous mettons en colonne les lignes du tableau
colonne = ['Zone', 'Y2017']
df_data = OCR.ligneToColonne(df_data, [2901, 664, 'Disponibilité alimentaire Kcal', colonne])
df_data = OCR.ligneToColonne(df_data, [2901, 674, 'Disponibilité alimentaire Protéine', colonne])
df_data = OCR.ligneToColonne(df_data, [2941, 674, 'Proportion Protéine Animale', colonne])
df_data = OCR.ligneToColonne(df_data, [2734, 664, 'Disponibilité alimentaire Kcal - volaille', colonne])
#df_data = OCR.ligneToColonne(df_data, [2734, 5511, 'Production de volaille', colonne])
df_data = OCR.ligneToColonne(df_data, [2734, 5611, 'Importation de volaille - Quantité', colonne])
df_data = OCR.ligneToColonne(df_data, [2734, 5911, 'Exportation de volaille - Quantité', colonne])

# Nous calculons la proportion de protéine animale par rapport au protéine totale
df_data['Proportion Protéine Animale'] /= df_data['Disponibilité alimentaire Protéine']

# Nous enlevons les lignes qui ne nous servent plus
filt = df_data['Code Élément'] == 511
colonne = ['Code zone', 'Zone', 'Difference de population', 'Disponibilité alimentaire Kcal', 'Disponibilité alimentaire Protéine', 'Proportion Protéine Animale', 'Y2017', 'Disponibilité alimentaire Kcal - volaille', 'Importation de volaille - Quantité', 'Exportation de volaille - Quantité']
df_data = df_data.loc[filt, colonne]

df_data.rename(columns={'Y2017': 'Population'}, inplace=True)

# Normalisation des variables de production et d'import/export
#df_data['Production de volaille'] /= df_data['Population']
df_data['Importation de volaille - Quantité'] /= df_data['Population'] 
df_data['Exportation de volaille - Quantité'] /= df_data['Population']

# Nous calculons la différence entre les importations et les exportations
df_data['Solde importation - Quantité'] = (df_data['Importation de volaille - Quantité'] - df_data['Exportation de volaille - Quantité']) * 1000

# ---------------------- Fichier Import Export Poulet ----------------------------------------------

# Nous importons le fichier d'importation et exportation de poulet
#df = OCR.NettoyagePays(pd.read_csv('./Data/data_import_export.csv'))

# Nous gardons que les lignes qui nous intéressent
#codeElementAGarder = [5610, 5910]
#codeProduitAGarder = [1061]

#df = OCR.ElementAGarder(df, codeElementAGarder, codeProduitAGarder)

# Nous mettons en colonne les lignes du tableau
#colonne = ['Zone', 'Valeur']
#df = OCR.ligneToColonne(df, [1061, 5610, 'Importations', colonne])
#df = OCR.ligneToColonne(df, [1061, 5910, 'Exportations', colonne])

# Nous calculons la différence entre les importations et les exportations
#df['Solde importation'] = df['Importations'] - df['Exportations']

# Nous enlevons les lignes qui ne nous servent plus
#filt = df['Code Élément'] == 5610
#colonne = ['Code zone', 'Solde importation']
#df = df.loc[filt, colonne]

# Nous mergeons les deux Data Frame
#df_data = df_data.merge(df, how = 'left', on = 'Code zone').fillna(df['Solde importation'].mean())

# Nous normalisons le solde d'importations par rapport à la population
#df_data['Solde importation'] /= df_data['Population']

# ---------------------- Fichier Elevage ---------------------------------------------------------

df = OCR.NettoyagePays(pd.read_csv('./Data/data_elevage.csv',encoding = "ISO-8859-1"))

# Nous gardons que les lignes qui nous intéressent
codeElementAGarder = [5510]
codeProduitAGarder = [1058]

df = OCR.ElementAGarder(df, codeElementAGarder, codeProduitAGarder)

# Nous enlevons les lignes qui ne nous servent plus
filt = df['Code Élément'] == 5510
colonne = ['Code zone', 'Y2017']
df = df.loc[filt, colonne]

# Nous mergeons les deux Data Frame
df_data = df_data.merge(df, how = 'left', on = 'Code zone').fillna(0)
df_data.rename(columns={'Y2017': 'Production'}, inplace=True)

# Nous normalisons le solde d'importations par rapport à la population
df_data['Production'] /= df_data['Population']

# ---------------------- Fichier Prix à la consommation ---------------------------------------------------------

#df = OCR.NettoyagePays(pd.read_csv('./Data/data_PAC.csv'))

# Nous gardons que les lignes qui nous intéressent
#df = df[df['Code Produit'] == 23012].groupby('Zone').mean()

# Nous enlevons les lignes qui ne nous servent plus
#df['Zone'] = df.index
#df.index = range(0, len(df))
#df = df[['Code zone', 'Valeur']]

# Nous mergeons les deux Data Frame
#df_data = df_data.merge(df, how = 'left', on = 'Code zone').fillna(df['Valeur'].mean())
#df_data.rename(columns={'Valeur': 'Prix Conso'}, inplace=True)

# ---------------------- Fichier PIB ---------------------------------------------------------

df = pd.read_csv('./Data/data_pib.csv')

# Nous mergeons les deux Data Frame
df_data = df_data.merge(df, how = 'left', on = 'Zone').fillna(0)

# ---------------------- Nettoyage du Fichier final ----------------------------------------------

# Nous enlevons les colonnes inutiles
colonne = ['Zone', 'Difference de population', 'Disponibilité alimentaire Kcal', 'Disponibilité alimentaire Protéine', 'Proportion Protéine Animale', 'Disponibilité alimentaire Kcal - volaille', 'Solde importation - Quantité', 'Production', 'PIB']
df_data = df_data[colonne]

# Nous exportons le fichier nettoyé
df_data.to_csv(path_or_buf='./Export/data.csv', index=False)
