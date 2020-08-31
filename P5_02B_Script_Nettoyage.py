import pandas as pd
import P5_03_OCR5 as OCR

df_data_raw = pd.read_csv('./Data/data_raw.csv',encoding = "ISO-8859-1")

# Nous enlevons les données concernant des groupements de pays (Code zone > 5000)
df_data = df_data_raw[~(df_data_raw['Code zone'] >= 5000)]

# Nous enlevons les données concernant la Chine qui est un groupement de pays (Code Zone = 351)
df_data = df_data[~(df_data['Code zone'] == 351)]

# Nous gardons que les lignes qui nous intéressent
codeElementAGarder = [511, 664, 674]
codeProduitAGarder = [2501, 2901, 2941]

df_data = df_data[df_data['Code Produit'].isin(codeProduitAGarder)]
df_data = df_data[df_data['Code Élément'].isin(codeElementAGarder)]

# Nous calculons la différence de populations en %
df_data['Difference de population'] = (df_data['Y2017'] - df_data['Y2014']) / df_data['Y2017'] * 100

# Nous mettons en colonne les lignes du tableau
colonne = ['Zone', 'Y2017']
df_data = OCR.ligneToColonne(df_data, [2901, 664, 'Disponibilité alimentaire Kcal', colonne])
df_data = OCR.ligneToColonne(df_data, [2901, 674, 'Disponibilité alimentaire Protéine', colonne])
df_data = OCR.ligneToColonne(df_data, [2941, 674, 'Proportion Protéine Animale', colonne])

# Nous calculons la proportion de protéine animale par rapport au protéine totale
df_data['Proportion Protéine Animale'] /= df_data['Disponibilité alimentaire Protéine']

# Nous enlevons les lignes qui ne nous servent plus
filt = df_data['Code Élément'] == 511
colonne = ['Zone', 'Difference de population', 'Disponibilité alimentaire Kcal', 'Disponibilité alimentaire Protéine', 'Proportion Protéine Animale']
df_data = df_data.loc[filt, colonne]

# Nous exportons le fichier nettoyé
df_data.to_csv(path_or_buf='./Export/data.csv', index=False)




