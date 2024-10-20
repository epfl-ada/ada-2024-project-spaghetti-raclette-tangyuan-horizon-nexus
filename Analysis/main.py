import inspect
import os
import pandas as pd

from pretraitement.pretraitement import pretraitement
from traitement.traitement import traitement
from analyse.analyse import analyse

# # Load the raw data #################################(Make it a function)
nom_fichier = 'donnees/character.metadata.tsv'
chemin_courant = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
chemin_complet = os.path.join(chemin_courant, nom_fichier)
data = pd.read_csv(chemin_complet, sep='\t')

# # Pre-treatment of data #######################(missing data, double entries, warnings)
data = pretraitement(data)

# # Treatment of data #######################(discret value transformation, add name of colomn and save names of discrete elements)
data = traitement(data)

# # Analysis
data = analyse(data)

print(data)