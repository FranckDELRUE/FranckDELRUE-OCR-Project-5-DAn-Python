#!/usr/bin/env python

import sys
import pandas as pd
import P5_03_OCR5 as OCR

# Création du DataFrame avec ajout des variables identifiées
df_raw = pd.read_csv(sys.argv[1])

# Nous enlevons les données concernant des groupements de pays (Code zone > 5000)
df = df_raw[~(df_raw['Code zone'] >= 5000)]

# Nous enlevons les données concernant la Chine qui est un groupement de pays (Code Zone = 351)
df = df[~(df['Code zone'] == 351)]

# Nous gardons que les lignes qui nous intéressent
codeElementAGarder = df['Code Élément'].unique()
codeProduitAGarder = df['Code Produit'].unique()

# Nous mettons en colonne les lignes du tableau
for element in codeElementAGarder:
    for produit in codeProduitAGarder:
        df = OCR.ligneToColonne(df_data, [element, produit, 'Disponibilité alimentaire Kcal'])

# Nous exportons le fichier nettoyé
df.to_csv(path_or_buf='./Export/data_test.csv', index=False)




