import numpy as np

from ._Epsilon import Epsilon


class ReinforcementComparison(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso del algoritmo de
    comparación de refuerzo (reinforcement comparison)
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    alpha : float
        Parámetro con el peso a usar para la actualización de la recompensa
    beta : floar
        Parámetro entre 0 y 1 que representa la tasa de aprendizaje
        
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
    Richard S. Sutton and Andrew G. Barto. "Reinfocement Learning: An
    Introduction". MIT Press, 1998.
    """

    def __init__(self, bandits, alpha=0.001, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        
        self._pi = [0] * len(bandits)
        self._r = [0] * len(bandits)
       
        super(ReinforcementComparison, self).__init__(bandits)
        
        
    def update(self, bandit, reward):
        self._r[bandit] = (1 - self.alpha) * self._r[bandit] + self.alpha * reward
        self._pi[bandit] += self.beta * (reward - self._r[bandit])

    
    def select(self):
        # Calculo de la probabilidad de seleccionar un bandido
        prob = np.exp(self._pi)
        prob /= np.sum(prob)
        prob = np.cumsum(prob)
             
        # Selección del bandido
        return np.where(prob > np.random.random())[0][0]