import numpy as np
import matplotlib.pyplot as plt


class Epsilon:
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia Epsilon
    Greedy
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    epsilon : float
        Porcentaje de veces en las que el agente jugada de forma
        aleatoria
    decay : float
        Velocidad con la que decae la probabilidad de seleccionar una
        jugada al azar
    initial: array of float
        Valor inicial de la recompensa esperada para cada uno de
        bandidos
        
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
    
    def __init__(self, bandits, epsilon=0.05, decay=1, initial=None):
        self.bandits = bandits
        self.epsilon = epsilon
        self.decay = decay
        
        self._num_bandits = len(bandits)
        self._rewards = []
        
        if initial is None:
            self._epsilon = self.epsilon
            self._plays = [0] * self._num_bandits
            self._mean = [0] * self._num_bandits
        else:
            self._epsilon = 0
            self._plays = [1] * self._num_bandits
            self._mean = initial
        
        
    def run(self, episodes=1):
        for i in range(episodes):
            # Selección del bandido
            bandit = self.select()
            
            # Obtención de una nueva recompensa
            reward = self.bandits[bandit].pull()
            
            # Agregación de la recompensa al listado
            self._rewards.append(reward)
            
            # Actualización de la media
            self._plays[bandit] += 1
            self._mean[bandit] = (1 - 1.0/self._plays[bandit]) * self._mean[bandit] \
                                 + 1.0/self._plays[bandit] * reward
            
            # Actualiza otros valores
            self.update(bandit, reward)
        
        return self.average_reward()
    
    
    def update(self, bandit, reward):
        pass
    
    
    def select(self):
        prob = np.random.random()
            
        # Selección entre la jugada aleatoria o avariciosa
        if prob < self._epsilon:
            bandit = np.random.choice(self._num_bandits)
        else:
            max_bandits = np.where(self._mean == np.max(self._mean))[0]
            bandit = np.random.choice(max_bandits)
        
        # Decaimiento del parámetro epsilon
        self._epsilon *= self.decay
        
        return bandit
    
                
    def average_reward(self):
        return np.mean(self._rewards)
    
    
    def plot(self, log=False, reference=False, label=None):
        cumulative_average = np.cumsum(self._rewards) / (np.arange(len(self._rewards)) + 1)
        
        if label is None:
            plt.plot(range(len(self._rewards)), cumulative_average)
        else:
            plt.plot(range(len(self._rewards)), cumulative_average, label=label)
            
        if reference:
            for reward in [b.reward for b in self.bandits]:
                plt.plot([0, len(self._rewards)], [reward, reward],
                         label=f'reward={reward}')
                
        if log:
            plt.xscale('log')