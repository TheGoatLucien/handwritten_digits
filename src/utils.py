"""Fonctions utiles pour les paliers 2 (MLP) et 3 (CNN) : jour 4, fiche 15.

- build_mlp    : construit un MLP Keras paramétrable
- plot_history : trace les courbes loss / erreur (entraînement et validation)
- run          : fait un essai complet (graine, compilation, entraînement chronométré,
                 courbes, ligne dans le tableau de résultats)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import tensorflow as tf
from tensorflow import keras


def build_mlp(hidden=(256,), dropout=0.0, batchnorm=False):
    """MLP pour des images aplaties (784 pixels) et 10 classes.

    hidden    : nombre de neurones de chaque couche cachée, ex. (256,) ou (512, 256)
    dropout   : taux de dropout après chaque couche cachée (0 = pas de dropout)
    batchnorm : True pour ajouter une BatchNormalization avant chaque ReLU
    """
    model = keras.Sequential(name="mlp")
    model.add(keras.Input(shape=(784,)))
    for n in hidden:
        if batchnorm:
            # Dense sans activation -> BatchNorm -> ReLU
            model.add(keras.layers.Dense(n))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("relu"))
        else:
            model.add(keras.layers.Dense(n, activation="relu"))
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


def plot_history(history, titre, save=None):
    """Deux graphiques côte à côte : la loss et l'erreur (1 - accuracy), train et validation."""
    h = history.history
    epochs = np.arange(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    axes[0].plot(epochs, h["loss"], label="entraînement")
    axes[0].plot(epochs, h["val_loss"], label="validation")
    axes[0].set_title("Loss")

    axes[1].plot(epochs, 100 * (1 - np.array(h["accuracy"])), label="entraînement")
    axes[1].plot(epochs, 100 * (1 - np.array(h["val_accuracy"])), label="validation")
    axes[1].set_title("Erreur (%)")

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs entières sur l'axe
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(titre)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100, bbox_inches="tight")
    plt.show()


def run(nom, creer_modele, optimizer, donnees, resultats, batch_size=128, epochs=20,
        callbacks=None, changement="", seed=42, save=None):
    """Un essai complet. Renvoie (model, history).

    creer_modele : une FONCTION qui construit le modèle, ex. lambda: build_mlp((256,))
                   -> on fixe la graine AVANT de construire, donc chaque essai part
                   des mêmes poids, et on repart toujours d'un modèle neuf.
    optimizer    : un optimiseur NEUF, ex. keras.optimizers.Adam(1e-3)
    donnees      : (x_train, y_train, x_val, y_val). Jamais le test ici.
    resultats    : la liste où on ajoute une ligne (un dict) par essai
    changement   : ce qui change par rapport à la référence (pour le tableau)
    save         : chemin où sauvegarder les courbes (optionnel)
    """
    x_train, y_train, x_val, y_val = donnees

    # Graine + opérations déterministes : relancer le même essai redonne les mêmes chiffres
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = creer_modele()
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    debut = time.perf_counter()
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, verbose=2)
    duree = time.perf_counter() - debut

    plot_history(history, nom, save=save)

    erreurs_val = 100 * (1 - np.array(history.history["val_accuracy"]))
    # Erreur du modèle FINAL sur la validation. Sans callback, c'est la valeur de la
    # dernière epoch ; avec EarlyStopping(restore_best_weights=True), ce sont les
    # meilleurs poids restaurés (et pas ceux de la dernière epoch).
    _, acc_val_finale = model.evaluate(x_val, y_val, verbose=0)
    resultats.append({
        "nom": nom,
        "changement": changement,
        "parametres": model.count_params(),
        "epochs": len(erreurs_val),  # peut être < epochs avec un EarlyStopping
        "erreur_val_%": round(100 * (1 - acc_val_finale), 2),
        "meilleure_erreur_val_%": round(float(erreurs_val.min()), 2),
        "epoch_meilleure": int(erreurs_val.argmin()) + 1,
        "duree_s": round(duree, 1),
    })
    return model, history
