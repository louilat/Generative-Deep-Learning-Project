'''Class of RBM model'''

import numpy as np


def lire_alpha_digits(data, list_index):
    m, n = data.shape
    p, q = data[0, 0].shape
    matrix = np.zeros((len(list_index)*n, p*q))
    for i, index in enumerate(list_index):
        for j in range(n):
            matrix[i*n + j] = data[index, j].reshape(p*q)
    return matrix


class RbmModel():
    def __init__(self, q: int, p: int) -> None:
        """
        Initialise un modèle de Restricted Boltzmann Machine (RBM).

        Args:
            q (int): Nombre de neurones dans la couche visible.
            p (int): Nombre de neurones dans la couche cachée.
        """
        self.q = q
        self.p = p
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(0, 0.1, size=(p, q))

    def entree_sortie_RBM(self, H: np.ndarray) -> np.ndarray:
        """
        Calcule l'activation de la couche visible à partir de l'activation de
        la couche cachée.

        Args:
            H (np.ndarray): Matrice d'activations de la couche cachée.

        Returns:
            np.ndarray: Activation de la couche visible.
        """
        A = np.tile(self.a, (H.shape[0], 1))
        Z = np.transpose(self.W @ H.T) + A
        return np.exp(Z)/(1 + np.exp(Z))

    def sortie_entree_RBM(self, V: np.ndarray) -> np.ndarray:
        """
        Calcule l'activation de la couche cachée à partir de l'activation de
        la couche visible.

        Args:
            V (np.ndarray): Matrice d'activations de la couche visible.

        Returns:
            np.ndarray: Activation de la couche cachée.
        """
        B = np.tile(self.b, (V.shape[0], 1))
        Z = np.transpose(self.W.T @ V.T) + B
        return np.exp(Z)/(1 + np.exp(Z))

    def split_in_batches(
            self, indexes: np.ndarray, batch: int) -> list([np.ndarray]):
        """
        Divise une liste d'indices en lots de la taille spécifiée.

        Args:
            indexes (list): Liste des indices à diviser en lots.
            batch (int): Taille de chaque lot.

        Returns:
            list: Liste des lots d'indices.
        """
        Batches = []
        N = len(indexes) // batch
        for i in range(N):
            Batches.append(indexes[i*batch:i*batch+batch])
        Batches.append(indexes[N*batch:len(indexes)])
        return Batches

    def train_RBM(
            self, X: np.ndarray, niter: int, step: float,
            batch: int, verbose: bool = True) -> None:
        """
        Entraîne la RBM avec les données d'entrée spécifiées.

        Args:
            X (np.ndarray): Données d'entrée pour l'entraînement.
            niter (int): Nombre d'itérations d'entraînement.
            step (float): Taux d'apprentissage.
            batch (int): Taille de lot pour l'entraînement par lot.
            verbose (bool): Affiche ou non les détails de l'entraînement.
        """
        assert X.shape[1] == self.W.shape[0]
        V = X.copy()
        batches_index = self.split_in_batches(np.arange(X.shape[0]), batch)
        rng = np.random.default_rng()
        for epoch in range(niter):
            loss = 0
            for batch in batches_index:
                X_batch = X[batch]
                Ph = self.sortie_entree_RBM(X_batch)
                H = rng.binomial(1, Ph, size=Ph.shape)
                Pv = self.entree_sortie_RBM(H)
                V = rng.binomial(1, Pv, size=Pv.shape)
                Mh = self.sortie_entree_RBM(V)
                loss += np.sum((V - X_batch)**2)

                # Compute the gradients
                grad_W = X_batch.T @ Ph - V.T @ Mh
                grad_a = np.sum(X_batch - V, axis=0)
                grad_b = np.sum(Ph - Mh, axis=0)

                # Update the parameters
                self.W += step*grad_W
                self.a += step*grad_a
                self.b += step*grad_b
            loss /= X.shape[0]
            if verbose:
                print("Epoch {}/{} : MSE = {}".format(epoch + 1, niter, loss))

    def generer_image_RBM(self, n_images: int, n_iter: int) -> np.ndarray:
        """
        Génère des images en utilisant le RBM.

        Args:
            n_images (int): Nombre d'images à générer.
            n_iter (int): Nombre d'itérations pour la génération.

        Returns:
            np.ndarray: Matrice des images générées.
        """
        X = np.random.normal(0, 0.1, size=(n_images, self.W.shape[0]))
        rng = np.random.default_rng()
        for _ in range(n_iter):
            Ph = self.sortie_entree_RBM(X)
            H = rng.binomial(1, Ph, size=Ph.shape)
            Pv = self.entree_sortie_RBM(H)
            X = rng.binomial(1, Pv, size=Pv.shape)
        return X
