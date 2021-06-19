import numpy as np

from ._Epsilon import Epsilon


class UCB1(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia UCB1
    
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
            ucb = [0] * self._num_bandits
            
            for i in range(self._num_bandits):
                ucb[i] = self._mean[i] + np.sqrt(2 * np.log(total) / self._plays[i])
        
            max_bandits = np.where(ucb == np.max(ucb))[0]
            bandit = np.random.choice(max_bandits)
            
        return bandit


class UCB2(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia UCB2
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    alpha : float
        Parámetro que se influye en el ratio de aprendizaje del algoritmo
    
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

    def __init__(self, bandits, alpha=0.1):
        self.alpha = alpha
            
        self._mean = [0] * len(bandits)
        
        super(UCB2, self).__init__(bandits)
    
    
    def select(self):
        total = len(self._rewards)
        
        if total == 0:
            bandit = np.random.choice(self._num_bandits)
        else:
            ucb = [0] * num_bandits
            
            for i in range(num_bandits):
                try:
                    tau = int(np.ceil((1 + self.alpha) ** self._plays[i]))
                    if np.log(np.e * total / tau) > 0:
                        bonus = np.sqrt((1. + self.alpha) * np.log(np.e * total / tau) / (2 * tau))
                    else:
                        bonus = 0
                except:
                    bonus = 0
                    
                if np.isnan(bonus):
                    ucb[i] = self._mean[i] 
                else:
                    ucb[i] = self._mean[i] + bonus
        
            max_bandits = np.where(ucb == np.max(ucb))[0]
            bandit = np.random.choice(max_bandits)
            
        return bandit


class UCB1Tuned(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia UCB1-Tuned
    
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

    def __init__(self, bandits):
        self._mean2 = [0] * len(bandits)
        
        super(UCB1Tuned, self).__init__(bandits)
    
    
    def update(self, bandit, reward):
        # Actualización de la media de los cuadrados
        self._mean2[bandit] = (1 - 1.0/self._plays[bandit]) * self._mean2[bandit] \
                              + 1.0/self._plays[bandit] * reward ** 2
            
            
    def select(self):
        total = len(self._rewards)
        
        if total == 0:
            bandit = np.random.choice(self._num_bandits)
        else:
            ucb = [0] * self._num_bandits
            
            for i in range(self._num_bandits):
                if self._plays[i] == 0:
                    v = self._mean2[i] - self._mean[i] ** 2 + np.sqrt(2 * np.log(total))
                else:
                    v = self._mean2[i] - self._mean[i] ** 2 + np.sqrt(2 * np.log(total) / self._plays[i])
        
                ucb[i] = self._mean[i] + np.sqrt(np.log(total) * np.min([1/4, v]))
            
            max_bandits = np.where(ucb == np.max(ucb))[0]
            bandit = np.random.choice(max_bandits)
            
        return bandit


class UCBNormal(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia UCB-Normal
    
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

    def __init__(self, bandits):
        self._rewards2 = [0] * len(bandits)
        
        super(UCBNormal, self).__init__(bandits)
        
        
    def update(self, bandit, reward):
        self._rewards2[bandit] += reward ** 2
        
        
    def select(self):
        total = len(self._rewards)
        
        # Número de veces mínimo que debe jugar cada bandido
        if total > 0:
            min_plays = np.ceil(8 * np.log(total))
        else:
            min_plays = 1
        
        # En caso de que algún bandido no jugase el mínimo de veces se selecciona ese
        if np.any(np.array(self._plays) < min_plays):
            min_bandit = np.where(np.array(self._plays) < min_plays)[0]
            bandit = np.random.choice(min_bandit)
        else:
            ucb = [0] * self._num_bandits
            
            for i in range(self._num_bandits):
                if self._plays[i] > 1:
                    bonus = 16 * (self._rewards2[i] - self._plays[i] * self._mean[i]**2) / (self._plays[i] - 1)
                    bonus *= np.log(total - 1) / self._plays[i]
                    bonus = np.sqrt(bonus)
                    ucb[i] = self._mean[i] + bonus
                else:
                    ucb[i] = self._mean[i]
                    
            max_bandits = np.where(ucb == np.max(ucb))[0]
            bandit = np.random.choice(max_bandits)
            
        return bandit


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
    Jean Yves Audibert, Rémi Munos, and Csaba Szepesvári.
    "Exploration-exploitation trade-off using variance estimates in multi-armed
     bandits." Theoretical Computer Science, Volume 410, Issue 19, 28 April 2009,
    Pages 1876-1902 (https://doi.org/10.1016/j.tcs.2009.01.016)
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
            