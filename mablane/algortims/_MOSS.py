import numpy as np

from ._Epsilon import Epsilon


class MOSS(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia MOSS
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
        
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

    def select(self):
        total = len(self._rewards)
        
        if total < self._num_bandits:
            bandit = total
        else:
            moss = [0] * self._num_bandits
            
            for i in range(self._num_bandits):
                moss[i] = self._mean[i] + np.sqrt(max(0, np.log(total/(self._num_bandits * self._plays[i]))) / self._plays[i])
        
            max_bandits = np.where(moss == np.max(moss))[0]
            bandit = np.random.choice(max_bandits)
            
        return bandit