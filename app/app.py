"""Application Flask (jour 12, fiche 22).

À écrire :
  - charger models/best.keras UNE fois, au démarrage
  - route "/"        : affiche templates/index.html (formulaire d'envoi)
  - route "/predict" : valide l'image (présente, lisible, 28 x 28), la prépare comme à
                       l'entraînement, renvoie le chiffre prédit + les 10 probabilités
                       (erreur -> code 400 + message clair)
"""
