'''Class of DNN model with Adam algorithm for both pretraining and training'''

import numpy as np


class DnnModel():
    def __init__(self, d: int, p: int, q: int, n_classes: int) -> None:
        """
        Initialise une instance de la classe DnnModel.

        Args:
            d (int): Nombre de couches dans le DNN.
            p (int): Dimension de la couche d'entrée.
            q (int): Dimension de la couche cachée.
            n_classes (int): Nombre de classes pour la classification.
        """
        self.d = d
        self.q = q
        self.p = p
        self.n_classes = n_classes
        self.A = np.zeros((d-1, q))
        self.B = np.zeros((d-1, q))
        self.W = np.random.normal(0, 0.1, size=(d-1, q, q))
        self.W0 = np.random.normal(0, 0.1, size=(p, q))
        self.a0 = np.zeros(p)
        self.b0 = np.zeros(q)
        # Classification
        self.Wc = np.random.normal(0, 0.1, size=(q, n_classes))
        self.bc = np.zeros(n_classes)

        # Moments
        self.mom_A = np.zeros((d-1, q))
        self.mom_B = np.zeros((d-1, q))
        self.mom_W = np.zeros((d-1, q, q))
        self.mom_W0 = np.zeros((p, q))
        self.mom_a0 = np.zeros(p)
        self.mom_b0 = np.zeros(q)
        # Classification
        self.mom_Wc = np.zeros((q, n_classes))
        self.mom_bc = np.zeros(n_classes)

        self.v_A = np.zeros((d-1, q))
        self.v_B = np.zeros((d-1, q))
        self.v_W = np.zeros((d-1, q, q))
        self.v_a0 = np.zeros(p)
        self.v_W0 = np.zeros((p, q))
        self.v_b0 = np.zeros(q)
        # Classification
        self.v_Wc = np.zeros((q, n_classes))
        self.v_bc = np.zeros(n_classes)

        self.accuracies_train = None
        self.accuracies_test = None
        self.losses_train = None

    def entree_sortie_RBM(self, d: int, H: np.ndarray) -> np.ndarray:
        """
        Calcule l'activation d'une couche d'un RBM. (dir. latente -> visible)

        Args:
            d (int): Indice de la couche RBM.
            H (np.ndarray): Matrice des activations de la couche
            précédente (d-1).
        """
        if d != 0:
            A = np.tile(self.A[d-1, :], (H.shape[0], 1))
            Z = np.transpose(self.W[d-1, :, :] @ H.T) + A
            return np.exp(Z)/(1 + np.exp(Z))
        else:
            A = np.tile(self.a0, (H.shape[0], 1))
            Z = np.transpose(self.W0 @ H.T) + A
            return np.exp(Z)/(1 + np.exp(Z))

    def sortie_entree_RBM(self, d: int, V: np.ndarray) -> np.ndarray:
        """
        Calcule l'activation d'une couche d'un RBM. (dir. visible -> latente)

        Args:
            d (int): Indice de la couche RBM.
            H (np.ndarray): Matrice des activations de la couche
            précédente (d-1).
        """
        if d != 0:
            B = np.tile(self.B[d-1, :], (V.shape[0], 1))
            Z = np.transpose(self.W[d-1, :, :].T @ V.T) + B
            return np.exp(Z)/(1 + np.exp(Z))
        else:
            B = np.tile(self.b0, (V.shape[0], 1))
            Z = np.transpose(self.W0.T @ V.T) + B
            return np.exp(Z)/(1 + np.exp(Z))

    def entree_sortie_DBN(self, H: np.ndarray) -> np.ndarray:
        """
        Calcule l'activation de la couche visible d'un Deep Belief
        Network (DBN)
        en utilisant les données de la couche latente.

        Args:
            H (np.ndarray): Matrice des activations de la couche cachée.

        Returns:
            np.ndarray: Activation de la couche visible.
        """
        rng = np.random.default_rng()
        E = H.copy()
        for t in range(1, self.d + 1):
            Pe = self.entree_sortie_RBM(self.d - t, E)
            E = rng.binomial(1, Pe, size=Pe.shape)
        return E

    def sortie_entree_DBN(self, V: np.ndarray) -> np.ndarray:
        """
        Calcule l'activation de la couche latente d'un Deep Belief
        Network (DBN)
        en utilisant les données de la couche visible.

        Args:
            V (np.ndarray): Matrice des activations de la couche cachée.

        Returns:
            np.ndarray: Activation de la couche latente.
        """
        rng = np.random.default_rng()
        S = V.copy()
        for t in range(self.d):
            Ps = self.sortie_entree_RBM(t, S)
            S = rng.binomial(1, Ps, size=Ps.shape)
        return S

    def classif_DNN(self, V: np.ndarray) -> tuple([np.ndarray, np.ndarray]):
        """
        Effectue la classification en utilisant les activations de la couche
        cachée du DNN.

        Args:
            V (np.ndarray): Matrice des activations de la dernière
            couche cachée.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Valeurs et probabilités après
            softmax de la classification.
        """
        S = V.copy()
        B = np.tile(self.bc, (V.shape[0], 1))
        Z = S @ self.Wc + B
        # Values
        # Values = np.exp(Z)/(1 + np.exp(Z))
        # Probas
        vector_sum = np.sum(np.exp(Z), axis=1)
        matrix_sum = np.tile(vector_sum, (Z.shape[1], 1)).T
        Probas = np.exp(Z) / matrix_sum
        return Z, Probas

    def split_in_batches(
            self, indexes: np.ndarray, batch: int) -> list([np.ndarray]):
        """
        Divise une liste d'indices en lots de la taille spécifiée.

        Args:
            indexes (np.ndarray): Liste des indices à diviser en lots.
            batch (int): Taille de chaque lot.

        Returns:
            List[np.ndarray]: Liste des lots d'indices.
        """
        Batches = []
        N = len(indexes) // batch
        for i in range(N):
            Batches.append(indexes[i*batch:i*batch+batch])
        Batches.append(indexes[N*batch:len(indexes)])
        return Batches

    def zero_mom(self):
        # Moments
        self.mom_A = np.zeros((self.d-1, self.q))
        self.mom_B = np.zeros((self.d-1, self.q))
        self.mom_W = np.zeros((self.d-1, self.q, self.q))
        self.mom_W0 = np.zeros((self.p, self.q))
        self.mom_a0 = np.zeros(self.p)
        self.mom_b0 = np.zeros(self.q)
        # Classification
        self.mom_Wc = np.zeros((self.q, self.n_classes))
        self.mom_bc = np.zeros(self.n_classes)

        self.v_A = np.zeros((self.d-1, self.q))
        self.v_B = np.zeros((self.d-1, self.q))
        self.v_W = np.zeros((self.d-1, self.q, self.q))
        self.v_W0 = np.zeros((self.p, self.q))
        self.v_a0 = np.zeros(self.p)
        self.v_b0 = np.zeros(self.q)
        # Classification
        self.v_Wc = np.zeros((self.q, self.n_classes))
        self.v_bc = np.zeros(self.n_classes)

    def train_RBM(
            self, t: int, X: np.ndarray, niter: int, gamma: float = 1e-3,
            beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
            batch: int = 256, verbose: bool = True) -> None:
        """
        Entraîne une couche RBM du réseau en utilisant la
        rétro-propagation contrastive (CD-k).

        Args:
            t (int): Indice de la couche RBM à entraîner.
            X (np.ndarray): Données d'entraînement.
            niter (int): Nombre d'itérations d'entraînement.
            step (float): Taux d'apprentissage pour la mise à jour des poids.
            batch (int): Taille de lot pour l'entraînement par lot.
            verbose (bool): Afficher ou non les détails de l'entraînement.
        """
        if t != 0:
            assert X.shape[1] == self.W[t-1, :, :].shape[0], (
                'Dimention of input incompatible with parameters of RBM')
        else:
            assert X.shape[1] == self.W0.shape[0], (
                'Dimention of input incompatible with parameters of RBM')
        self.zero_mom()
        batches_index = self.split_in_batches(np.arange(X.shape[0]), batch)
        rng = np.random.default_rng()
        for epoch in range(niter):
            loss = 0
            for batch in batches_index:
                X_batch = X[batch]
                N = X_batch.shape[0]
                Ph = self.sortie_entree_RBM(t, X_batch)
                H = rng.binomial(1, Ph, size=(N, Ph.shape[1]))
                Pv = self.entree_sortie_RBM(t, H)
                V = rng.binomial(1, Pv, size=(N, Pv.shape[1]))
                Mh = self.sortie_entree_RBM(t, V)
                loss += np.sum((V - X_batch)**2)

                # Compute the gradients
                grad_W = X_batch.T @ Ph - V.T @ Mh
                grad_a = np.sum(X_batch - V, axis=0)
                grad_b = np.sum(Ph - Mh, axis=0)

                # Update the parameters
                if t != 0:
                    self.mom_W[t-1, :, :] = beta1 * self.mom_W[t-1, :, :] + (
                        1 - beta1) * grad_W
                    self.mom_B[t-1, :] = beta1 * self.mom_B[t-1, :] + (
                        1 - beta1) * grad_b
                    self.mom_A[t-1, :] = beta1 * self.mom_A[t-1, :] + (
                        1 - beta1) * grad_a
                    self.v_W[t-1, :, :] = beta2 * self.v_W[t-1, :, :] + (
                        1 - beta2) * grad_W**2
                    self.v_B[t-1, :] = beta2 * self.v_B[t-1, :] + (
                        1 - beta2) * grad_b**2
                    self.v_A[t-1, :] = beta2 * self.v_A[t-1, :] + (
                        1 - beta2) * grad_a**2

                    mom_W_hat = (self.mom_W[t-1, :, :] / (
                        1 - beta1**(epoch + 1))).astype(np.float64)
                    mom_A_hat = (self.mom_A[t-1, :] / (
                        1 - beta1**(epoch + 1))).astype(np.float64)
                    mom_B_hat = (self.mom_B[t-1, :] / (
                        1 - beta1**(epoch + 1))).astype(np.float64)
                    v_W_hat = (self.v_W[t-1, :, :] / (
                        1 - beta2**(epoch + 1))).astype(np.float64)
                    v_A_hat = (self.v_A[t-1, :] / (
                        1 - beta2**(epoch + 1))).astype(np.float64)
                    v_B_hat = (self.v_B[t-1, :] / (
                        1 - beta2**(epoch + 1))).astype(np.float64)

                    self.W[t-1, :, :] += gamma * mom_W_hat / (
                        np.sqrt(v_W_hat) + epsilon)
                    self.A[t-1, :] += gamma * mom_A_hat / (
                        np.sqrt(v_A_hat) + epsilon)
                    self.B[t-1, :] += gamma * mom_B_hat / (
                        np.sqrt(v_B_hat) + epsilon)
                else:
                    self.mom_W0 = beta1 * self.mom_W0 + (1 - beta1) * grad_W
                    self.mom_b0 = beta1 * self.mom_b0 + (1 - beta1) * grad_b
                    self.mom_a0 = beta1 * self.mom_a0 + (1 - beta1) * grad_a
                    self.v_W0 = beta2 * self.v_W0 + (1 - beta2) * grad_W**2
                    self.v_b0 = beta2 * self.v_b0 + (1 - beta2) * grad_b**2
                    self.v_a0 = beta2 * self.v_a0 + (1 - beta2) * grad_a**2

                    mom_W0_hat = (self.mom_W0 / (
                        1 - beta1**(epoch + 1))).astype(np.float64)
                    mom_a0_hat = (self.mom_a0 / (
                        1 - beta1**(epoch + 1))).astype(np.float64)
                    mom_b0_hat = (self.mom_b0 / (
                        1 - beta1**(epoch + 1))).astype(np.float64)
                    v_W0_hat = self.v_W0 / (1 - beta2**(epoch + 1))
                    v_a0_hat = self.v_a0 / (1 - beta2**(epoch + 1))
                    v_b0_hat = self.v_b0 / (1 - beta2**(epoch + 1))

                    self.W0 += gamma * mom_W0_hat / (
                        np.sqrt(v_W0_hat) + epsilon)
                    self.a0 += gamma * mom_a0_hat / (
                        np.sqrt(v_a0_hat) + epsilon)
                    self.b0 += gamma * mom_b0_hat / (
                        np.sqrt(v_b0_hat) + epsilon)
            loss /= X.shape[0]
            if verbose:
                print("Epoch {}/{} : MSE = {}".format(epoch + 1, niter, loss))

    def pretrain_DNN(
        self, X: np.ndarray, niter: int, batch: int,
        gamma: float = 1e-3, beta1: float = 0.9,
        beta2: float = 0.999, epsilon: float = 1e-8,
            verbose: bool = True, full_verbose: bool = False) -> None:
        """
        Pré-entraîne les RBM du DNN en cascade avec les données
        d'entrée spécifiées.

        Args:
            X (np.ndarray): Données d'entrée pour l'entraînement préalable.
            niter (int): Nombre d'itérations d'entraînement par RBM.
            step (float): Taux d'apprentissage.
            batch (int): Taille de lot pour l'entraînement par lot.
            verbose (bool): Afficher ou non les détails de l'entraînement.
        """
        S = X.copy()
        rng = np.random.default_rng()
        for t in range(self.d):
            if verbose:
                print('Training Layer {}/{} ...'.format(t+1, self.d))
            self.train_RBM(
                t=t, X=S, niter=niter, batch=batch, gamma=gamma,
                beta1=beta1, beta2=beta2, epsilon=epsilon,
                verbose=full_verbose)
            if verbose:
                Ph = self.sortie_entree_RBM(t, S)
                H = rng.binomial(1, Ph, size=Ph.shape)
                Pv = self.entree_sortie_RBM(t, H)
                V = rng.binomial(1, Pv, size=Pv.shape)
                loss = np.sum((V - S)**2) / X.shape[0]
                print('Reconstruction error (MSE) for layer {} = {}'.format(
                    t+1, loss))
            Ps = self.sortie_entree_RBM(t, S)  # CHANGE HERE
            S = rng.binomial(1, Ps, size=(X.shape[0], self.q))

    def generer_image_DBN(self, n_images: int, n_iter: int) -> np.ndarray:
        """
        Génère des images en utilisant le Deep Belief Network (DBN).

        Args:
            n_images (int): Nombre d'images à générer.
            n_iter (int): Nombre d'itérations pour la génération.

        Returns:
            np.ndarray: Matrice des images générées.
        """
        X = np.random.binomial(1, 0.5, size=(n_images, self.q))
        E = X.copy()
        rng = np.random.default_rng()
        for _ in range(n_iter):
            Pv = self.entree_sortie_DBN(E)
            V = rng.binomial(1, Pv, size=(n_images, self.p))
            Ph = self.sortie_entree_DBN(V)
            E = rng.binomial(1, Ph, size=(n_images, self.q))
        return V

    def sortie_entree_RBM_retro(
            self, d: int, V: np.ndarray) -> tuple([np.ndarray, np.ndarray]):
        """
        Calcule l'activation de la couche cachée en fonction de l'activation
        de la couche visible
        pour une couche RBM spécifique (utilisé lors de la rétro-propagation).

        Args:
            d (int): Indice de la couche RBM.
            V (np.ndarray): Matrice des activations de la couche visible.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Valeurs et probabilités d'
            activation pour la couche cachée.
        """
        if d != 0:
            # print(self.W[d-1, :, :].T.shape)
            # print(V.T.shape)
            B = np.tile(self.B[d-1, :], (V.shape[0], 1))
            Z = np.transpose(self.W[d-1, :, :].T @ V.T) + B
            return Z, np.exp(Z)/(1 + np.exp(Z))
        else:
            B = np.tile(self.b0, (V.shape[0], 1))
            Z = np.transpose(self.W0.T @ V.T) + B
            return Z, np.exp(Z)/(1 + np.exp(Z))

    def entree_sortie_reseau(
            self, X: np.ndarray) -> tuple(
                [list([np.ndarray]), list([np.ndarray])]):
        """
        Effectue la propagation avant à travers le réseau et retourne les
        valeurs et probabilités d'activation pour chaque couche.

        Args:
            X (np.ndarray): Données d'entrée.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Valeurs et probabilités
            d'activation pour chaque couche.
        """
        Values = []
        Probas = []
        A = X.copy()
        for t in range(self.d):
            Z, A = self.sortie_entree_RBM_retro(t, A)
            Values.append(Z)
            Probas.append(A)
        S, P = self.classif_DNN(A)
        Values.append(S)
        Probas.append(P)
        return Values, Probas

    def retropropagation(
            self, X: np.ndarray, y: np.ndarray, t: int, gamma: float = 1e-3,
            beta1: float = 0.9, beta2: float = 0.999,
            epsilon: float = 1e-8) -> None:
        """
        Effectue la rétro-propagation à travers le réseau pour
        ajuster les poids.

        Args:
            X (np.ndarray): Données d'entrée pour la rétro-propagation.
            y (np.ndarray): Étiquettes de classe correspondantes.
            epsilon (float): Taux d'apprentissage pour la mise à jour des
            poids.
        """
        N = X.shape[0]
        # One hot encoding of y
        Y = np.eye(self.n_classes)[y]
        # Forward pass
        Values, Probas = self.entree_sortie_reseau(X)
        # Backward pass sur couche de classification
        # print('Training classif...')
        A2 = Probas[-1].copy()
        A1 = Values[-2].copy()
        grad_Lz = A2 - Y
        grad_LW = 1/N * A1.T @ grad_Lz
        grad_Lb = grad_Lz.mean(axis=0)
        grad_La2 = grad_Lz @ self.Wc.T

        self.mom_Wc = beta1 * self.mom_Wc + (1 - beta1) * grad_LW
        self.mom_bc = beta1 * self.mom_bc + (1 - beta1) * grad_Lb
        self.v_Wc = beta1 * self.v_Wc + (1 - beta2) * grad_LW**2
        self.v_bc = beta1 * self.v_bc + (1 - beta2) * grad_Lb**2
        mom_Wc_hat = self.mom_Wc / (1 - beta1**(t+1))
        mom_bc_hat = self.mom_bc / (1 - beta1**(t+1))
        v_Wc_hat = self.v_Wc / (1 - beta2**(t+1))
        v_bc_hat = self.v_bc / (1 - beta2**(t+1))

        # Updates classif layer
        self.Wc -= gamma * mom_Wc_hat / (np.sqrt(v_Wc_hat) + epsilon)
        self.bc -= gamma * mom_bc_hat / (np.sqrt(v_bc_hat) + epsilon)

        # Backward pass sur le DBN
        # print('Classif trained')
        for d in range(2, self.d + 1):
            # print('layer number {} '.format(d))
            A2 = Probas[-d].copy()
            A1 = Probas[-(d+1)].copy()
            # print('Shapes A1, A2 = {}, {}'.format(A1.shape, A2.shape))
            sig_p = np.multiply(A2, 1-A2)
            grad_Lz = np.multiply(grad_La2, sig_p)
            grad_LW = 1/N * A1.T @ grad_Lz
            grad_Lb = grad_Lz.mean(axis=0)
            grad_La2 = grad_Lz @ self.W[-(d-1), :, :].T
            if d == self.d + 1:
                self.mom_W0 = beta1 * self.mom_W0 + (1 - beta1) * grad_LW
                self.mom_b0 = beta1 * self.mom_b0 + (1 - beta1) * grad_Lb
                self.v_W0 = beta1 * self.v_W0 + (1 - beta2) * grad_LW**2
                self.v_b0 = beta1 * self.v_b0 + (1 - beta2) * grad_Lb**2
                mom_W0_hat = self.mom_W0 / (1 - beta1**(t+1))
                mom_b0_hat = self.mom_b0 / (1 - beta1**(t+1))
                v_W0_hat = self.v_W0 / (1 - beta2**(t+1))
                v_b0_hat = self.v_b0 / (1 - beta2**(t+1))

                self.W0 -= gamma * mom_W0_hat / (np.sqrt(v_W0_hat) + epsilon)
                self.b0 -= gamma * mom_b0_hat / (np.sqrt(v_b0_hat) + epsilon)
            else:
                self.mom_W[-(d-1), :, :] = beta1 * self.mom_W[-(d-1), :, :] + (
                    1 - beta1) * grad_LW
                self.mom_B[-(d-1), :] = beta1 * self.mom_B[-(d-1), :] + (
                    1 - beta1) * grad_Lb
                self.v_W[-(d-1), :] = beta1 * self.v_W[-(d-1), :] + (
                    1 - beta2) * grad_LW**2
                self.v_B[-(d-1), :] = beta1 * self.v_B[-(d-1), :] + (
                    1 - beta2) * grad_Lb**2
                mom_W_hat = self.mom_W[-(d-1), :, :] / (1 - beta1**(t+1))
                mom_b_hat = self.mom_B[-(d-1), :] / (1 - beta1**(t+1))
                v_W_hat = self.v_W[-(d-1), :] / (1 - beta2**(t+1))
                v_b_hat = self.v_B[-(d-1), :] / (1 - beta2**(t+1))
                # Updates RBM layers
                self.W[-(d-1), :, :] -= gamma * mom_W_hat / (
                    np.sqrt(v_W_hat) + epsilon)
                self.B[-(d-1), :] -= gamma * mom_b_hat / (
                    np.sqrt(v_b_hat) + epsilon)

    def train_DNN(
            self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
            y_test: np.ndarray, n_epochs: int, gamma: float = 1e-3,
            beta1: float = 0.9, beta2: float = 0.999, batch: int = 256,
            epsilon: float = 1e-8,
            training_track: bool = False, verbose: bool = True
            ) -> None:
        """
        Entraîne le réseau de neurones profonds (DNN) en utilisant
        la rétro-propagation.

        Args:
            X_train (np.ndarray): Données d'entraînement.
            y_train (np.ndarray): Étiquettes de classe correspondantes pour
            l'entraînement.
            X_test (np.ndarray): Données de test.
            y_test (np.ndarray): Étiquettes de classe correspondantes pour les
            tests.
            n_epochs (int): Nombre d'époques d'entraînement.
            epsilon (float): Taux d'apprentissage pour les mises à jour des
            poids.
            batch (int): Taille de lot pour l'entraînement par lot.
            training_track (bool): Suivre ou non les précisions d'entraînement
            et de test
                (par défaut False).
            verbose (bool): Afficher ou non les détails de l'entraînement (par
            défaut True).

        Returns:
            None

        Example:
            dnn = DnnModel()
            X_train, y_train, X_test, y_test = load_data()
            dnn.train_DNN(X_train, y_train, X_test, y_test,
                n_epochs=50, epsilon=0.01, batch=32, training_track=True)
        """
        if verbose or training_track:
            Accuracies_train = []
            Accuracies_test = []
            Losses_train = []
            Y_train = np.eye(self.n_classes)[y_train]
        self.zero_mom()
        batches_index = self.split_in_batches(
            np.arange(X_train.shape[0]), batch)
        for epoch in range(n_epochs):
            for batch in batches_index:
                X_batch = X_train[batch]
                y_batch = y_train[batch]
                self.retropropagation(
                    X=X_batch, y=y_batch, t=epoch,
                    gamma=gamma, beta1=beta1, beta2=beta2,
                    epsilon=epsilon)
            if verbose or training_track:
                score_train = self.predict_score(X_train, y_train)
                score_test = self.predict_score(X_test, y_test)
                _, probas = self.entree_sortie_reseau(X_train)
                Y_probas = probas[-1]
                loss_train = - np.mean(
                    np.multiply(Y_train, np.log(Y_probas)).sum(axis=1))
                if verbose:
                    print(''.join([
                        'Epoch {}/{} : Accuracy score train',
                        ' = {} | Accuracy score test = {}']).format(
                            epoch, n_epochs, score_train, score_test))
                if training_track:
                    Accuracies_train.append(score_train)
                    Accuracies_test.append(score_test)
                    Losses_train.append(loss_train)
        if training_track:
            self.accuracies_train = Accuracies_train
            self.accuracies_test = Accuracies_test
            self.losses_train = Losses_train

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les classes pour les données d'entrée spécifiées.

        Args:
            X (np.ndarray): Données d'entrée pour les prédictions.

        Returns:
            np.ndarray: Prédictions de classe.
        """
        _, probas = self.entree_sortie_reseau(X)
        y_hat = probas[-1]
        return np.argmax(y_hat, axis=1)

    def predict_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcule le score d'accuracy pour les données d'entrée spécifiées.

        Args:
            X (np.ndarray): Données d'entrée pour évaluer les prédictions.
            y (np.ndarray): Étiquettes de classe correspondantes.

        Returns:
            float: Score de précision.
        """
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
