# Fine-tuning BERT model for question answering with SQUAD dataset / DISTRIBUTED

## Colaborators
- Omar Montenegro Macía
- Álvaro Pardo Fente

## Table of contents
Nothing here...

## Profiling outputs

### Execution times

### Tensorboard

Las diferencias entre el entrenamiento distribuido realizado y el entrenamiento secuencial del modelo en BASELINE residen en esta nueva vista llamada **Distributed**. Como el nombre indica, esta vista otorga información relacionada con el entrenamiento distribuido realizado. En esta primera tabla podemos ver la información del hardware usado en el entrenamiento, se muestran 2 procesos por cada uno de los 2 nodos, cada proceso usando una GPU **NVIDIA A100**.

![image](https://github.com/user-attachments/assets/757b9326-c593-424c-b18b-eda52a5cc512)

En la siguiente captura podemos ver la generación de 2 gráficas. Comenzando por la de la izquierda, esta muestra el tiempo total de la ejecución por step desglosado en diferentes categorías:
- Computación: la suma de las ejecuciones de cada kernel de la GPU **menos** el tiempo de **solapamiento**.
- Comunicación: el tiempo total de comunicación **menos** el tiempo de **solapamiento**.
- Solapamiento: el tiempo total en el que el tiempo de computación y el de comunicación **se han solapado**. Un mayor valor de este tiempo implica un **mejor paralelismo** en la ejecución del entrenamiento, ya que implica un decremento en el tiempo de computación y en el de comunicación **de forma simultánea**. Idealmente, el tiempo de solapamiento **cubriría totalmente** el tiempo de comunicación (en la captura puede verse como esta situación prácticamente se logra).
- Otro: tiempo del step **menos** el tiempo de computación y el de comunicación (relacionado con el tiempo de computación de la CPU, carga de los dataset en DataLoaders, entre otros factores).

TODO: Esta gráfica da una visión clara acerca de los tiempos de ejecución de cada worker...

TODO: Continuando con la gráfica de la derecha, su objetivo es mostrar la eficiencia de las comunicaciones, es decir...

Por último, se muestra un panel con todos los detalles relevantes acerca de la ejecución de las **funciones de comunicación usadas por worker**. Se muestra el nº de llamadas a dichas funciones, el tamaño de los mensajes compartidos, la latencia asociada a la transmisión y el tiempo de transferencia de los datos.
- En la captura, se puede observar la ejecución de 2 funciones colectivas (broadcast y all_reduce) de la API **NCCL**, que es la implementación de NVIDIA de MPI.

![image](https://github.com/user-attachments/assets/70576066-0bc8-4747-b4bb-0d56efd772ec)
