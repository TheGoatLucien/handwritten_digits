"""Chargement et découpage de MNIST, partagé par les 3 notebooks (jour 2, fiche 3).

À écrire : une fonction load_data() qui
  - charge MNIST avec keras.datasets.mnist.load_data()
  - découpe les 60 000 images en 54 000 (train) / 6 000 (validation), stratifié, random_state=42
  - divise les pixels par 255
  - renvoie x_train, y_train, x_val, y_val, x_test, y_test
"""
