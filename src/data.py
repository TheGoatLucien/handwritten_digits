"""Chargement et découpage de MNIST, partagé par les 3 notebooks (jour 2, fiche 3)."""

from sklearn.model_selection import train_test_split
from tensorflow import keras


def load_data(seed=42):
    """Charge MNIST et renvoie les trois jeux : entraînement, validation, test.

    - Les 60 000 images d'entraînement de MNIST sont découpées en
      54 000 (entraînement) / 6 000 (validation), de façon stratifiée
      pour garder la même proportion de chaque chiffre dans les deux jeux.
    - Les 10 000 images de test ne sont pas touchées (elles servent à la fin).
    - Les pixels passent de 0-255 (entiers) à 0-1 (float32).
    - Les images restent en 28 x 28 : chaque notebook les remet en forme
      selon le modèle (vecteur de 784 pour le MLP, 28 x 28 x 1 pour le CNN).

    Paramètres
    ----------
    seed : int
        Graine du découpage, pour retrouver toujours le même.

    Renvoie
    -------
    x_train, y_train, x_val, y_val, x_test, y_test
    """
    (x_full, y_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=6000, stratify=y_full, random_state=seed
    )

    x_train = x_train.astype("float32") / 255
    x_val = x_val.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    return x_train, y_train, x_val, y_val, x_test, y_test
