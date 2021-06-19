import numpy as np

from statsmodels.stats.proportion import proportion_confint

from ._Epsilon import Epsilon


def klBin(p, q, n=1, eps=1e-15):
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    
    return n * (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


class KLUCB(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia KL-UCB
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    n : float
        El número de pruebas de la distribución binomial
    c : float
        Parámetro con el que se puede modificar la velocidad de aprendizaje
        
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
    Aurélien Garivier and Olivier Cappé. "The KL-UCB Algorithm for Bounded
    Stochastic Bandits and Beyond." arXiv preprint arXiv:1102.2490 (2011).

    Giuseppe Burtini, Jason Loeppky, and Ramon Lawrence. "A survey of online
    experiment design with the stochastic multi-armed bandit." arXiv preprint
    arXiv:1510.00757 (2015).
    """

    def __init__(self, bandits, n=1, c=0):
        self.n = n
        self.c = c
        
        super(KLUCB, self).__init__(bandits)
        
    
    def select(self):
        total = len(self._rewards)
        
        if total < self._num_bandits:
            bandit = total
        else:
            ucb = [0] * self._num_bandits
            d = np.log(total) + self.c * np.log((total + 1))
                
            for i in range(self._num_bandits):
                ucb[i] = klBin(self._mean[i], d / self._plays[i], self.n)
        
            max_bandits = np.where(ucb == np.max(ucb))[0]
            bandit = np.random.choice(max_bandits)
            
        return bandit


class CPUCB(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia CP-UCB
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    c : float
        Parámetro con el que se puede modificar la velocidad de aprendizaje
    method : string
        Método empleado para calcular el intervalo de confianza con la función
        proportion_confint de statsmodels
        
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
    Aurélien Garivier and Olivier Cappé. "The KL-UCB Algorithm for Bounded
    Stochastic Bandits and Beyond." arXiv preprint arXiv:1102.2490 (2011).
    """

    def __init__(self, bandits, c=1, method='beta'):
        self.c = c
        self.method = method
        
        self._reward = [0] * len(bandits)
        
        super(CPUCB, self).__init__(bandits)
        
    
    def update(self, bandit, reward):
        self._reward[bandit] += reward
        
        
    def select(self):
        total = len(self._rewards)
        
        if total < self._num_bandits:
            bandit = total
        else:
            ucb = [0] * self._num_bandits
                
            for i in range(self._num_bandits):
                confidence = 1 / (total * np.log(total) ** self.c)
                ucb[i] = proportion_confint(self._reward[i], self._plays[i], confidence, method=self.method)[1]            
            
            max_bandits = np.where(ucb == np.max(ucb))[0]
            bandit = np.random.choice(max_bandits)
            
        return bandit