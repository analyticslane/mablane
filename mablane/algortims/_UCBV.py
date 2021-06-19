import numpy as np

from ._Epsilon import Epsilon


class UCBV(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia UCBV
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    b : float
        Hiperparámetro para seleccionar el ration de aprendizaje
        
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

    def __init__(self, bandits, b=3):
        self.b = b
        
        self._mean2 = [0] * len(bandits)
        
        super(UCBV, self).__init__(bandits)
        
    
    def update(self, bandit, reward):
        self._mean2[bandit] += reward**2
        
        
    def select(self):
        num_bandits = len(self.bandits)
        total = len(self._rewards)
        
        if total < num_bandits:
            bandit = total
        else:
            ucb = [0] * num_bandits
                
            for i in range(num_bandits):
                var = self._mean2[i] / self._plays[i] - self._mean[i]**2
                ucb[i] = self._mean[i]
                ucb[i] += np.sqrt(2 * var * np.log(total) / self._plays[i])
                ucb[i] += self.b * np.log(total) / self._plays[i]             
            
            max_bandits = np.where(ucb == np.max(ucb))[0]
            bandit = np.random.choice(max_bandits)
            