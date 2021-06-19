from numpy.random import negative_binomial


class NegativeBinomialBandit:
    """
    Implementación de un Bandido Multibrazo (Multi-Armed Bandit) basado
    en una distribución binomial negativa

    Parámetros
    ----------
    number: integer
        Número de recompensas que puede devolver el agente
    probability : float
        Probabilidad de que el objeto devuelva una recompensa
    
    Métodos
    -------
    pull :
        Realiza una tirada en el bandido
        
    """
    def __init__(self, probability, number=1):
        self.number = number
        self.probability = probability
        
        self.reward = self.number * self.probability
        
        
    def pull(self):
        """ Realiza una tirada en el bandido

        Retorna
        -------
        reward: float
            Recompensa obtenida en la tirada
        """  
        return negative_binomial(self.number, self.probability)