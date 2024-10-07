# Fine-tuning BERT model for question answering with SQUAD dataset / DISTRIBUTED

## Colaborators
- Omar Montenegro Macía
- Álvaro Pardo Fente

## Table of contents
- [How to run?](#how-to-run)
- [DDP](#ddp)
- [Profiling outputs](#profiling-outputs)
  - [Execution times](#execution-times)
 	- [Tensorboard](#tensorboard)

## How to run?
Como en el BASELINE, para crear el venv de python y ejecutar el entrenamiento distribuido lanzamos el siguiente script:

~~~shell
./launch.sh
~~~

## DDP
La estrategia elegida para el entrenamiento distribuido ha sido **DDP**, ya que es una estrategia sencilla y efectiva **si el modelo entero cabe en 1 sola GPU**. En DDP los nodos se dividen los batches que conforman el dataset y los procesan por varias iteraciones del modelo completo. Para obtener el total global de los pesos calculados por cada worker, la implementación de DDP en pytorch_lightning usa la estrategia **mirrored**, que utiliza operaciones de comunicación colectivas como **all-reduce**.
- Las desventajas de esta técnica son las **continuas sincronizaciones entre los workers** (lo que consume gran parte del tiempo de comunicaciones) y la **poca escalabilidad**, ya que se debe de incluir el **modelo entero** en la memoria de cada worker, lo que limita el tamaño del modelo + el tamaño del batch a la memoria disponible en la GPU.
  - DDP puede trabajar con el paralelismo **a nivel de modelo**, lo que permite superar la mencionada limitación.

## Profiling outputs

### Execution times

<table><tr><th colspan="1"></th><th colspan="1"><b>Optimizers</b></th><th colspan="1"><b>Step 0</b></th><th colspan="1"><b>Step 1</b></th><th colspan="1"><b>Step 2</b></th><th colspan="1"><b>Step 3</b></th><th colspan="1"><b>Step 4</b></th><th colspan="1"><b>Step 5</b></th><th colspan="1"><b>Avg step</b></th><th colspan="1"><b>Avg epoch (150 steps)</b></th><th colspan="1"><b>Total exc</b></th></tr>
<tr><td colspan="1" rowspan="2">Sequential</td><td colspan="1">AdamW</td><td colspan="1">2607,112 ms</td><td colspan="1">2115,140 ms</td><td colspan="1">2106,955 ms</td><td colspan="1">2107,439 ms</td><td colspan="1">2109,853 ms</td><td colspan="1">2110,829 ms</td><td colspan="1">2192,888 ms</td><td colspan="1">5,48222 mins</td><td colspan="1">33,2294 mins</td></tr>
<tr><td colspan="1">SGD</td><td colspan="1">2505,104 ms</td><td colspan="1">2145,509 ms</td><td colspan="1">2135,011 ms</td><td colspan="1">2138,431 ms</td><td colspan="1">2140,558 ms</td><td colspan="1">2142,426 ms</td><td colspan="1">2201,173 ms</td><td colspan="1">5,5029325 mins</td><td colspan="1">33,0592 mins</td></tr>
<tr><td colspan="1" rowspan="2">DDP</td><td colspan="1">AdamW</td><td colspan="1"><a name="content6"></a>2713,093 ms</td><td colspan="1"><a name="content7"></a>2225,034 ms</td><td colspan="1"><a name="content8"></a>2171,767 ms</td><td colspan="1">2190,840 ms</td><td colspan="1"><a name="content10"></a>2165,545 ms</td><td colspan="1"><a name="content11"></a>2,184.132 ms</td><td colspan="1">2275,06ms</td><td colspan="1">5,68765 min</td><td colspan="1">32 mins??</td></tr>
<tr><td colspan="1">SGD</td><td colspan="1"><a name="content"></a>2962,157 ms</td><td colspan="1"><a name="content1"></a>2143,035 ms</td><td colspan="1"><a name="content2"></a>2142,375 ms</td><td colspan="1"><a name="content3"></a>2135,773 ms</td><td colspan="1"><a name="content4"></a>2138,628 ms</td><td colspan="1"><a name="content5"></a>2140,032 ms</td><td colspan="1">2276,995 ms</td><td colspan="1">5,69 mins</td><td colspan="1">8,7387 mins (tiempo secuencial/4)</td></tr>
</table>

Los tiempos de step en la versión DDP son contando **sólo uno de los workers**.

Puede apreciarse un ligero aumento del tiempo de ejecución en la versión distribuida, que se podría atribuir al aumento del número de las comunicaciones entre los workers al intercambiar datos. Sin embargo, en el tiempo final se refleja un notable decremento, concretamente se establece esta relación:

$Tiempo DDP = Tiempo Secuencial/num Workers$

### Tensorboard

Las diferencias entre el entrenamiento distribuido realizado y el entrenamiento secuencial del modelo en BASELINE residen en esta nueva vista llamada **Distributed**. Como el nombre indica, esta vista otorga información relacionada con el entrenamiento distribuido realizado. En esta primera tabla podemos ver la información de la jerarquía de los nodos, procesos y GPUs usadas en el entrenamiento, se muestran 2 procesos por cada uno de los 2 nodos, cada proceso usando una GPU **NVIDIA A100**.

![image](https://github.com/user-attachments/assets/757b9326-c593-424c-b18b-eda52a5cc512)

En la siguiente captura podemos ver la generación de 2 gráficas. Comenzando por la de la izquierda, esta muestra el tiempo total de la ejecución por step desglosado en diferentes categorías:
- Computación: la suma de las ejecuciones de cada kernel de la GPU **menos** el tiempo de **solapamiento**.
- Comunicación: el tiempo total de comunicación **menos** el tiempo de **solapamiento**.
- Solapamiento: el tiempo total en el que el tiempo de computación y el de comunicación **se han solapado**. Un mayor valor de este tiempo implica un **mejor paralelismo** en la ejecución del entrenamiento, ya que implica un decremento en el tiempo de computación y en el de comunicación **de forma simultánea**. Idealmente, el tiempo de solapamiento **cubriría totalmente** el tiempo de comunicación (en la captura puede verse como esta situación prácticamente se logra).
- Otro: tiempo del step **menos** el tiempo de computación y el de comunicación (relacionado con el tiempo de computación de la CPU, carga de los dataset en DataLoaders, entre otros factores).

Atendiendo de nuevo a la captura, podemos ver que los tiempos de computación, comunicación + solapamiento entre los 4 workers están equilibrados (hay un correcto balanceo de la carga).

Continuando con la gráfica de la derecha, su objetivo es mostrar la eficiencia de las **comunicaciones**, es decir, que del tiempo total cuanto se invierte en cada una de las siguientes operaciones:
- Transferencia de datos: tiempo invertido en intercambiar datos entre los workers.
- Sincronización: duración de la espera de los workers por datos que necesitan de otros workers o a que estos terminen para continuar con su ejecución desde un punto de sincronización.

En el caso de la captura, la gran mayoría del tiempo se invierte en la sincronización entre los workers, lo que indica una **ineficiencia** en las comunicaciones. Como ya se mencionó, el entrenamiento distribuido utiliza la estrategia **mirrored**, que usa funciones de comunicación colectivas para propagar los cambios de los pesos. Estas funciones establecen **puntos de sincronización**. Una mejora sería realizar estas sincronizaciones con menos frecuencia (lo que afecta a la precisión del modelo, **TODO**: probarla) o intercambiar los gradients.

Por último, se muestra un panel con todos los detalles relevantes acerca de la ejecución de las **funciones de comunicación usadas por worker**. Se muestra el nº de llamadas a dichas funciones, el tamaño de los mensajes compartidos, la latencia asociada a la transmisión y el tiempo de transferencia de los datos.
- En la captura, se puede observar la ejecución de 2 funciones colectivas (broadcast y all_reduce) de la API **NCCL**, que es la implementación de NVIDIA de MPI. 

![image](https://github.com/user-attachments/assets/70576066-0bc8-4747-b4bb-0d56efd772ec)

TODO: Si volvemos a la vista de memoria que analizamos previamente en el BASELINE, podemos ver que el consumo de memoria de la GPU sigue igual.
