import numpy as np

from scipy.stats import beta

from ._Epsilon import Epsilon


class ThompsonSampling(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso del Muestreo de Thompson
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    N : float
        El número de sucesos de la distribución Binomial
        
    Métodos
    -------
    run :
        Realiza una serie de tiradas con los bandidos seleccionados
        por el algoritmo
    update:
        Actualiza los valores adicionales después de una tirada
    select :
        Selecciona un bandido para jugar en la próxima tirada
    average_reward :
        Obtención de la recompensa promedio
    plot :
        Representación gráfica del histórico de tiradas

    References
    ----------
    Emilie Kaufmann, Nathaniel Korda, and Rémi Munos. "Thompson Sampling: An
    Asymptotically Optimal Finite Time Analysis." arXiv preprint
    arXiv:1205.4217 (2012).
    """

    def __init__(self, bandits, N=1):
        self.N = N
        
        self._alpha = [1] * len(bandits)
        self._beta = [1] * len(bandits)
            
        super(ThompsonSampling, self).__init__(bandits)
        
        
    def update(self, bandit, reward):
        # Guardado de valores intermedos
        self._alpha[bandit] += reward
        self._beta[bandit] += (self.N - reward)
            
            
    def select(self):
        bayes = [0] * self._num_bandits
        
        for i in range(self._num_bandits):
            bayes[i] = beta.rvs(self._alpha[i], self._beta[i])
            
        max_bandits = np.where(bayes == np.max(bayes))[0]
        bandit = np.random.choice(max_bandits)
        
        return bandit


class BayesUCB(ThompsonSampling):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia Bayes-UCB
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    N : float
        El número de sucesos de la distribución Binomial
    gamma : float
        Parámetro con el que se indica cuántas desviaciones
        estándar queremos para el nivel de confianza
        
    Métodos
    -------
    run :
        Realiza una serie de tiradas con los bandidos seleccionados
        por el algoritmo
    update:
        Actualiza los valores adicionales después de una tirada
    select :
        Selecciona un bandido para jugar en la próxima tirada
    average_reward :
        Obtención de la recompensa promedio
    plot :
        Representación gráfica del histórico de tiradas

    References
    ----------
    Giuseppe Burtini, Jason Loeppky, and Ramon Lawrence. "A survey of online
    experiment design with the stochastic multi-armed bandit." arXiv preprint
    arXiv:1510.00757 (2015).

    Emilie Kaufmann, Olivier Cappe, and Aurelien Garivier. "On Bayesian Upper
    Confidence Bounds for Bandit Problems." Proceedings of the Fifteenth
    International Conference on Artificial Intelligence and Statistics, PMLR
    22:592-600, 2012.
    """

    def __init__(self, bandits, N=1, gamma=3):
        self.gamma = gamma
        
        super(BayesUCB, self).__init__(bandits, N)

    
    def select(self):
        bayes = [0] * self._num_bandits
        
        for i in range(self._num_bandits):
            bayes[i] = self._mean[i] + beta.std(self._alpha[i], self._beta[i]) * self.gamma
            
        max_bandits = np.where(bayes == np.max(bayes))[0]
        bandit = np.random.choice(max_bandits)
            
        return bandit