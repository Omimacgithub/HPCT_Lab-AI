# Fine-tuning BERT model for question answering with SQUAD dataset / DISTRIBUTED

## Colaborators
- Omar Montenegro Macía
- Álvaro Pardo Fente

## Table of contents
- [How to run?](#how-to-run)
- [Explanation of the code](#explanation-of-the-code)
- [DDP](#ddp)
- [Profiling outputs](#profiling-outputs)
  - [Execution times](#execution-times)
  - [Tensorboard](#tensorboard)

## How to run?
Como en el BASELINE, para crear el venv de python y ejecutar el entrenamiento distribuido lanzamos el siguiente script:

~~~shell
./launch.sh
~~~

**Para ver los datos generados con tensorboard**:

~~~shell
source ./mypython/bin/activate
tensorboard --logdir=./l_runs --host `hostname -i` &
http://<IP_del_nodo>:6006/#pytorch_profiler
~~~

## Explanation of the code

Si el entrenamiento se encuentra implementado en pytorch_lightning, paralelizarlo mediante DDP es cuestión de poner 1 único parámetro nuevo a la clase trainer:

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/d091510cfc63e0aceca3c88931be21ad1eaac66c/DISTRIBUTED/lightning_training.py#L149

También es necesario controlar mediante 2 parámetros el nº de dispositivos y el nº de nodos que intervendrán en el entrenamiento (en la sección [Execution times](#execution-times) se explica más detalladamente):

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/d091510cfc63e0aceca3c88931be21ad1eaac66c/DISTRIBUTED/lightning_training.py#L145-L146

## DDP
La estrategia elegida para el entrenamiento distribuido ha sido **DDP**, ya que es una estrategia sencilla y efectiva **si el modelo entero + tamaño del batch cabe en 1 sola GPU**, además de que se encuentra implementado de **forma nativa** en pytorch_lightning. En DDP los nodos se dividen los batches que conforman el dataset y los procesan por varias iteraciones del modelo completo (paralelismo a nivel de datos) **cada uno de forma simultánea**. Para obtener el total global de los pesos calculados por cada worker, la implementación de DDP en pytorch_lightning usa la estrategia **mirrored**, que utiliza operaciones de comunicación colectivas como **all-reduce** para que todos los nodos obtengan el total global de estos pesos para continuar con la siguiente iteración del modelo.
- Las desventajas de esta técnica son las **continuas sincronizaciones entre los workers** (lo que consume gran parte del tiempo de comunicaciones) y la **poca escalabilidad**, ya que se debe de incluir el **modelo entero** en la memoria de cada worker, lo que limita el tamaño del modelo + el tamaño del batch a la memoria disponible en la GPU.
  - DDP puede trabajar con el paralelismo **a nivel de modelo**, lo que permite superar la mencionada limitación.

## Profiling outputs

### Execution times

Se recogen 3 salidas del entrenamiento de BERT para DDP usando SGD usando 1, 2 y 4 GPUs y las 2 salidas anteriormente vistas en BASELINE. Se usó la siguiente configuración:

- Fase de entrenamiento con 22500 filas del dataset de entrenamiento.
  - Tamaño de batch de 150
- Fase de validación con las 200 primeras filas del dataset de validación y test con las 200 últimas filas del dataset de validación.
  - Tamaño de batch de 8 (200/8 = 25 steps)
- 6 epochs.
- Semilla=42, **es importante inicializar una semilla para todos los workers, ya que sino puede dar lugar a pesos distintos entre modelos**.

Cambiar la cantidad de GPUs(devices) y de nodos(num_nodes) es tan sencillo como cambiar 2 parámetros de la clase Trainer que nos proporciona pytorch_lightning:

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/d091510cfc63e0aceca3c88931be21ad1eaac66c/DISTRIBUTED/lightning_training.py#L143-L146

- Además, es necesario cambiar los parámetros correspondientes al **job.sbatch**:

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/d091510cfc63e0aceca3c88931be21ad1eaac66c/DISTRIBUTED/job.sbatch#L2-L5

- Línea 2: indica el nº de nodos del supercomputador que intervendrán en el proceso (equivale al parámetro **num_nodes**).
- Línea 3: indica para cada nodo el nº de tareas a ejecutar (equivale al parámetro **devices**).
- Línea 5: para solicitar el nº de GPUs por nodo. **En el FT3, a excepción de unos pocos nodos, la mayoría tienen un máximo de 2 GPUs A100**.

<table><tr><th colspan="1"></th><th colspan="1"><b>Optimizers</b></th><th colspan="1"><b>Step 0</b></th><th colspan="1"><b>Step 1</b></th><th colspan="1"><b>Step 2</b></th><th colspan="1"><b>Step 3</b></th><th colspan="1"><b>Step 4</b></th><th colspan="1"><b>Step 5</b></th><th colspan="1"><b>Avg step</b></th><th colspan="1"><b>Avg epoch (150 steps)</b></th><th colspan="1"><b>Total exc</b></th></tr>
<tr><td colspan="1" rowspan="2">Sequential</td><td colspan="1">AdamW</td><td colspan="1">2607,112 ms</td><td colspan="1">2115,140 ms</td><td colspan="1">2106,955 ms</td><td colspan="1">2107,439 ms</td><td colspan="1">2109,853 ms</td><td colspan="1">2110,829 ms</td><td colspan="1">2192,888 ms</td><td colspan="1">5,48222 mins</td><td colspan="1">33,2294 mins</td></tr>
<tr><td colspan="1">SGD</td><td colspan="1">2505,104 ms</td><td colspan="1">2145,509 ms</td><td colspan="1">2135,011 ms</td><td colspan="1">2138,431 ms</td><td colspan="1">2140,558 ms</td><td colspan="1">2142,426 ms</td><td colspan="1">2201,173 ms</td><td colspan="1">5,5029325 mins</td><td colspan="1">33,0592 mins</td></tr>
<tr><td colspan="1" rowspan="2">DDP</td><td colspan="1">AdamW</td><td colspan="1"><a name="content6"></a>2713,093 ms</td><td colspan="1"><a name="content7"></a>2225,034 ms</td><td colspan="1"><a name="content8"></a>2171,767 ms</td><td colspan="1">2190,840 ms</td><td colspan="1"><a name="content10"></a>2165,545 ms</td><td colspan="1"><a name="content11"></a>2,184.132 ms</td><td colspan="1">2275,06ms</td><td colspan="1">5,68765 min</td><td colspan="1">32 mins??</td></tr>
<tr><td colspan="1">SGD</td><td colspan="1"><a name="content"></a>2962,157 ms</td><td colspan="1"><a name="content1"></a>2143,035 ms</td><td colspan="1"><a name="content2"></a>2142,375 ms</td><td colspan="1"><a name="content3"></a>2135,773 ms</td><td colspan="1"><a name="content4"></a>2138,628 ms</td><td colspan="1"><a name="content5"></a>2140,032 ms</td><td colspan="1">2276,995 ms</td><td colspan="1">5,69 mins</td><td colspan="1">8,7387 mins (tiempo secuencial/4)</td></tr>
</table>

Los tiempos de step en la versión DDP son contando **sólo uno de los workers**.

Puede apreciarse un ligero aumento del tiempo de ejecución en la versión distribuida, que se podría atribuir al aumento del número de las comunicaciones entre los workers al intercambiar datos. Sin embargo, en el tiempo final se refleja un notable decremento, concretamente se establece esta relación (**considerando el mismo tamaño del dataset y de los batches**):

$$ Tiempo DDP = \frac{Tiempo Secuencial}{Num Workers} $$

Es decir, la duracción del entrenamiento se reduce de forma **lineal** según el nº de workers disponibles, también se establece la siguiente relación:

$$ Steps Per Epoch = \frac{Filas Dataset}{Batch Size * Num Workers} $$

Siendo **NumWorkers = nº de nodos * nº de dispositivos (GPUs)**.

Acorde a esta relación, si aumentas el nº de workers que intervienen en el entrenamiento, menos steps tendrá cada epoch (22500/(150*4) = 37,5 steps), por lo tanto menos tiempo durará el proceso. Si aumentas el tamaño del dataset, más nº de steps habrá por epoch, por ende más durará el entrenamiento (por lo que se puede establecer un equilibrio entre aumentar el tamaño del dataset y el nº de los workers para mantener el tiempo de ejecución "estable"). El problema de DDP es que **no nos permite aumentar el tamaño del batch**, debido a que cada worker debe de albergar el **modelo entero + batch size en su GPU** (riesgo de un **CUDAOutOfMemory**).

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

Si volvemos a la vista de memoria que analizamos previamente en el BASELINE, cada worker consume la misma cantidad de memoria que en el entrenamiento secuencial con los mismos parámetros.

![image](https://github.com/user-attachments/assets/23516621-8095-42b1-9907-4c7b4dcd8737)


## Reassemble splited output files

Los archivos de salida de los entrenamientos ocupan en total en torno a unos ** 1,4 GB**:
- 3 salidas por pantalla del código de entrenamiento con 1, 2 y 4 GPUs (todas usan SGD como optimizador).
- Archivos de profiling de los 3 entrenamientos para examinar con tensorboard
- 1 checkpoint del entrenamiento final del modelo usando 4 GPUs (directorio version\_0).

Como el tamaño máximo de ficheros permitido por GitHub es de 100MB se han fragmentado los ficheros. Para recomponer los archivos de salida es necesario realizar los siguientes pasos:

Juntar los ficheros en 1 sólo zip.
~~~shell
cd outputs
cat outputsa? > outputs.zip
~~~

Descomprimir el fichero final.
~~~shell
unzip outputs.zip
~~~

Analizar los datos del profiler:

~~~shell
tensorboard --logdir=./outputs --host `hostname -i` &
~~~
