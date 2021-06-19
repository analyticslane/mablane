import numpy as np

from ._Epsilon import Epsilon


class Exp3(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia Exp3
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    gamma : float
        Probabilidad de seleccionar un bandido de forma aleatoria
        
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
    """

    def __init__(self, bandits, gamma=0.05):
        self.gamma = gamma
        
        self._weights = [1] * len(bandits) 
        
        super(Exp3, self).__init__(bandits)
        
            
    def update(self, bandit, reward):
        # Actualización de los pesos
        self._weights[bandit] *= np.exp(self._mean[bandit] * self.gamma / self._num_bandits)
        
        
    def select(self):
        total = len(self._rewards)
        
        if total < self._num_bandits:
            bandit = total
        else:
            exp3 = [0] * self._num_bandits
            
            total_weights = np.sum(self._weights)

            for i in range(self._num_bandits):
                exp3[i] = (1 - self.gamma) * self._weights[i] / total_weights + self.gamma / self._num_bandits
        
            bandit = np.random.choice(self._num_bandits, p=exp3)
            
        return bandit