import numpy as np

from ._Epsilon import Epsilon


class Pursuit(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de algoritmos de
    seguimiento (pursuit)
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    beta : floar
        Hiperparámetro entre 0 y 1 que representa la tasa de aprendizaje.
        
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

    def __init__(self, bandits, beta=0.01):
        self.beta = beta
        
        self._p = [1 / len(bandits)] * len(bandits)
        
        super(Pursuit, self).__init__(bandits)
        
       
    def update(self, bandit, reward):
        max_bandit = np.argmax(self._mean)
        
        for i in range(self._num_bandits):
            if i == max_bandit:
                self._p[i] += self.beta * (1 - self._p[i])
            else:
                self._p[i] -= self.beta * self._p[i]
    
    
    def select(self):
        # Calculo de la probabilidad de seleccionar un bandido
        prob = np.cumsum(self._p)
        
        # Selección del bandido
        return np.where(prob > np.random.random())[0][0] 