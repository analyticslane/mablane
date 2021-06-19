import numpy as np

from ._Epsilon import Epsilon

class Softmax(Epsilon):
    """
    Agente que soluciona el problema del el Bandido Multibrazo
    (Multi-Armed Bandit) mediante el uso de una estrategia Softmax
    
    Parámetros
    ----------
    bandits : array of Bandit
        Vector con los bandidos con los que se debe jugar
    tau : float
        Hiperparámetro empleado para seleccionar al bandido 
        
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
    """
    
    def __init__(self, bandits, tau=0.01):
        self.tau = tau
        
        super(Softmax, self).__init__(bandits)
        
    
    def select(self):
        # Calculo de la probabilidad de seleccionar un bandido
        prob = [np.exp(m / self.tau) for m in self._mean]
        prob /= np.sum(prob)
        prob = np.cumsum(prob)
        
        # Selección del bandido
        return np.where(prob > np.random.random())[0][0] 