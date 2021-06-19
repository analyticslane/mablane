El Bandido Multibrazo (Multi-Armed Bandit)
==========================================

Implementación de bandidos y diferentes algoritmos para resolver
problemas de tipo Bandido Multibrazo (Multi-Armed Bandit) en Python. Los
bandidos que se han implementado son:

-  binomial
-  binomial negativa

Los algoritmos que se pueden encontrar en la librería son:

-  `Algoritmos de seguimiento
   (pursuit) <https://www.analyticslane.com/2021/06/11/algoritmos-de-seguimiento-pursuit-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `Comparación de refuerzo (reinforcement
   comparison) <https://www.analyticslane.com/2021/06/18/comparacion-de-refuerzo-reinforcement-comparison-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `Softmax <https://www.analyticslane.com/2021/03/19/softmax-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `Epsilon-Greedy <https://www.analyticslane.com/2021/02/26/epsilon-gready-para-el-bandido-multibrazo-multi-armed-bandit/>`__
-  `Valores iniciales
   optimistas <https://www.analyticslane.com/2021/03/12/valores-iniciales-optimistas-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `Muestreo de
   Thompson <https://www.analyticslane.com/2021/04/30/muestreo-de-thompson-y-bayesucb-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `BayesUCB <https://www.analyticslane.com/2021/04/30/muestreo-de-thompson-y-bayesucb-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `UCB <https://www.analyticslane.com/2021/03/26/ucb1-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `UCB2 <https://www.analyticslane.com/2021/04/09/ucb2-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `UCB1-Tuned <https://www.analyticslane.com/2021/04/16/ucb1-tuned-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `UCB1-Normal <https://www.analyticslane.com/2021/04/23/ucb1-normal-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `KL-UCB <https://www.analyticslane.com/2021/05/07/kl-ucb-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `EXP3 <https://www.analyticslane.com/2021/05/14/exp3-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `MOSS <https://www.analyticslane.com/2021/05/21/moss-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `UCB_V <https://www.analyticslane.com/2021/05/28/ucb-v-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__
-  `CP-UCB <https://www.analyticslane.com/2021/06/04/cp-ucb-para-un-problema-bandido-multibrazo-multi-armed-bandit/>`__

Instalación
-----------

La instalación del paquete se puede hacer desde el repositorio de
github, para lo que solamente se tiene que escribir el siguiente
comando:

::

   pip install git+https://github.com/analyticslane/mablane.git

Ejemplo de uso
==============

Una vez instalado el paquete ya se puede usar para crear vector de
bandidos y un agente, para lo que simplemente se deben seguir los
siguientes pasos

::

   from mablane.bandits import BinomialBandit
   from mablane.algortims import UCB1

   # Creación de los bandidos
   bandits = [BinomialBandit(0.02), BinomialBandit(0.06), BinomialBandit(0.10)]

   # Creación del agente
   ucb1 = UCB1(bandits)

   # Simular 10000 lanzamientos
   ucb1.run(10000)

   # Recompensa promedio
   ucb1.average_reward()

   # Veces que ha jugado con cada bandido
   ucb1._plays

Disclaimer
----------

Copyright (c) 2021 Daniel Rodríguez Pérez

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
“Software”), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
