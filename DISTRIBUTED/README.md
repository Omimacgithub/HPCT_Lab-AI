# Fine-tuning BERT model for question answering with SQUAD dataset / DISTRIBUTED

## Colaborators
- Omar Montenegro Macía
- Álvaro Pardo Fente

## Table of contents
- [How to run?](#how-to-run)
- [Explanation of the code](#explanation-of-the-code)
- [Estrategia de distribución del entrenamiento](#estrategia-de-distribucion-del-entrenamiento)
- [Profiling outputs](#profiling-outputs)
  - [Execution times](#execution-times)
  - [Tensorboard](#tensorboard)
- [Reassemble splited output files](#reassemble-splited-output-files)

## Objective

Tras obtener un código de fine-tuning en Pytorch Lightning del modelo de BERT (BASELINE), el siguiente objetivo es **paralelizar** el entrenamiento para reducir los **33 minutos** obtenidos tras entrenar el modelo en 1 GPU NVIDIA A100. Para conseguir dicho objetivo, contamos con **2 nodos del Finisterrae III, cada uno con 2 GPUs NVIDIA A100**, en el que ejecutaremos una **estrategia de distribución del entrenamiento** para repartir el trabajo entre múltiples máquinas (o workers).

## Estrategia de distribución del entrenamiento
La estrategia elegida ha sido **Distributed Data Parallel (DDP)**, ya que es sencilla y efectiva **si el modelo entero + tamaño del batch cabe en 1 sola GPU**, además de que se encuentra implementado de **forma nativa** en Pytorch Lightning. En Distributed Data Parallel, los nodos se dividen el tamaño del dataset en batches del tamaño especificado y los procesan por varias iteraciones del modelo completo (paralelismo a nivel de datos) **cada uno de forma simultánea**. 

Para obtener la suma total de los pesos del modelo, la implementación de DDP en Pytorch Lightning usa la estrategia **mirrored**, que aplica operaciones de comunicación colectivas como **all-reduce** para comunicar los gradientes a todos los workers. Finalmente, cada worker invoca al optimizador para actualizar los pesos y seguir con la siguiente iteración del entrenamiento.

La principal limitación de DDP es que **no es escalable**, puesto a que se debe de incluir todos los parámetros del modelo y el batch de entrada en la memoria de cada worker (GPU en este caso). La estrategia **Fully Sharded Data Parallel (FSDP)** permite distribuir los parámetros del modelo entre los workers, lo que otorga una **mayor escalabilidad**. Por otro lado, el número de comunicaciones con FSDP **aumenta** respecto a DDP, aunque estas se pueden "ocultar" si se solapan con el tiempo de computación.

## Explanation of the code

Si el entrenamiento se encuentra implementado en pytorch_lightning, paralelizarlo mediante DDP es cuestión de poner 1 único parámetro nuevo a la clase trainer:

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/d091510cfc63e0aceca3c88931be21ad1eaac66c/DISTRIBUTED/lightning_training.py#L149

También es necesario controlar mediante 2 parámetros el nº de dispositivos y el nº de nodos que intervendrán en el entrenamiento (en la sección [Execution times](#execution-times) se explica más detalladamente):

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/d091510cfc63e0aceca3c88931be21ad1eaac66c/DISTRIBUTED/lightning_training.py#L145-L146

## How to run?
Como en el BASELINE, para crear el venv de python y ejecutar el entrenamiento distribuido lanzamos el siguiente script:

~~~shell
./launch.sh
~~~

**Para ver los datos generados con tensorboard**:

~~~shell
source ../mypython/bin/activate
tensorboard --logdir=./l_runs --host `hostname -i` &
http://<IP_del_nodo>:6006/#pytorch_profiler
~~~

## Profiling outputs

### Execution times

Se recogen 3 salidas del entrenamiento de BERT para DDP con el optimizador **SGD** usando 1, 2 y 4 GPUs y las 2 salidas anteriormente vistas en BASELINE. Se usó la siguiente configuración:

- Fase de entrenamiento con 22500 filas del dataset de entrenamiento.
  - Tamaño de batch de 150
- Fase de validación con las 200 primeras filas del dataset de validación y test con las 200 últimas filas del dataset de validación.
  - Tamaño de batch de 8 (200/8 = 25 steps)
- 6 epochs, la duración de un epoch **varía según el nº de los workers:**
  - 1 GPU: 150 steps
  - 2 GPUs: 75 steps (150/2)
  - 4 GPUs: 37,5 steps (150/4, 38 steps en 3 GPUs y 36 en 1 GPU)
- Semilla=42, **es importante inicializar una semilla para todos los workers, ya que sino puede dar lugar a pesos distintos entre modelos**.

Cambiar la cantidad de GPUs(devices) y de nodos(num_nodes) es tan sencillo como cambiar 2 parámetros de la clase Trainer que nos proporciona pytorch_lightning:

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/d091510cfc63e0aceca3c88931be21ad1eaac66c/DISTRIBUTED/lightning_training.py#L143-L146

- Además, es necesario cambiar los parámetros correspondientes al **job.sbatch**:

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/d091510cfc63e0aceca3c88931be21ad1eaac66c/DISTRIBUTED/job.sbatch#L2-L5

- Línea 2: indica el nº de nodos del supercomputador que intervendrán en el proceso (equivale al parámetro **num_nodes**).
- Línea 3: indica para cada nodo el nº de tareas a ejecutar (equivale al parámetro **devices**).
- Línea 5: para solicitar el nº de GPUs por nodo. **En el FT3, a excepción de unos pocos nodos, la mayoría tienen un máximo de 2 GPUs A100**.

<table width="1089" cellpadding="3" cellspacing="0"><col width="108"/><col width="100"/><col width="106"/><col width="88"/><col width="89"/><col width="88"/><col width="89"/><col width="88"/><col width="89"/><col width="90"/><col width="89"/><thead><tr><th width="108" style="border: none; padding: 0in"><p><br/></p></th><th width="100" style="border: none; padding: 0in"><p>Optimizers</p></th><th width="106" style="border: none; padding: 0in"><p>Step 0</p></th><th width="88" style="border: none; padding: 0in"><p>Step 1</p></th><th width="89" style="border: none; padding: 0in"><p>Step 2</p></th><th width="88" style="border: none; padding: 0in"><p>Step 3</p></th><th width="89" style="border: none; padding: 0in"><p>Step 4</p></th><th width="88" style="border: none; padding: 0in"><p>Step 5</p></th><th width="89" style="border: none; padding: 0in"><p>Avg step</p></th><th width="90" style="border: none; padding: 0in"><p>Avg epoch</p></th><th width="89" style="border: none; padding: 0in"><p>Total exc (6 epochs)</p></th></tr></thead><tbody><tr><td rowspan="2" width="108" style="border: none; padding: 0in"><p>Sequential</p></td><td width="100" style="border: none; padding: 0in"><p>AdamW</p></td><td width="106" style="border: none; padding: 0in"><p>2607,112 ms</p></td><td width="88" style="border: none; padding: 0in"><p>2115,140 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2106,955 ms</p></td><td width="88" style="border: none; padding: 0in"><p>2107,439 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2109,853 ms</p></td><td width="88" style="border: none; padding: 0in"><p>2110,829 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2192,888 ms</p></td><td width="90" style="border: none; padding: 0in"><p>5,48 mins</p></td><td width="89" style="border: none; padding: 0in"><p>33,2294 mins</p></td></tr><tr><td width="100" style="border: none; padding: 0in"><p>SGD</p></td><td width="106" style="border: none; padding: 0in"><p>2505,104 ms</p></td><td width="88" style="border: none; padding: 0in"><p>2145,509 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2135,011 ms</p></td><td width="88" style="border: none; padding: 0in"><p>2138,431 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2140,558 ms</p></td><td width="88" style="border: none; padding: 0in"><p>2142,426 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2201,173 ms</p></td><td width="90" style="border: none; padding: 0in"><p>5,50 mins</p></td><td width="89" style="border: none; padding: 0in"><p>33,0592 mins</p></td></tr><tr><td rowspan="3" width="108" height="39" style="border: none; padding: 0in"><p>DDP(SGD)</p></td><td width="100" style="border: none; padding: 0in"><p>1 GPU</p></td><td width="106" style="border: none; padding: 0in"><p style="background: transparent">2370,463 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2111,413 ms</p></td><td width="89" style="border: none; padding: 0in"><p style="background: transparent">2110,497 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2110,867 ms</p></td><td width="89" style="border: none; padding: 0in"><p style="background: transparent">2133,193 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2113,210 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2158,274 ms</p></td><td width="90" style="border: none; padding: 0in"><p>5,4 mins</p></td><td width="89" style="border: none; padding: 0in"><p>32,700 mins</p></td></tr><tr><td width="100" style="border: none; padding: 0in"><p>2 GPUs</p></td><td width="106" style="border: none; padding: 0in"><p style="background: transparent">2463,650 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2126,777 ms</p></td><td width="89" style="border: none; padding: 0in"><p style="background: transparent">2154,995 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2157,522 ms</p></td><td width="89" style="border: none; padding: 0in"><p style="background: transparent">2156,224 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2164,616 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2203,9715 ms</p></td><td width="90" style="border: none; padding: 0in"><p>2,75 mins</p></td><td width="89" style="border: none; padding: 0in"><p> 16,9767 mins</p></td></tr><tr><td width="100" style="border: none; padding: 0in"><p>4 GPUs</p></td><td width="106" style="border: none; padding: 0in"><p style="background: transparent">2962,157 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2143,035 ms</p></td><td width="89" style="border: none; padding: 0in"><p style="background: transparent">2142,375 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2135,773 ms</p></td><td width="89" style="border: none; padding: 0in"><p style="background: transparent">2138,628 ms</p></td><td width="88" style="border: none; padding: 0in"><p style="background: transparent">2140,032 ms</p></td><td width="89" style="border: none; padding: 0in"><p>2276,995 ms</p></td><td width="90" style="border: none; padding: 0in"><p>1,42 mins</p></td><td width="89" style="border: none; padding: 0in"><p>8,7387 mins</p></td></tr></tbody></table>

Los tiempos de step en la versión DDP son contando **sólo uno de los workers** (excepto en la versión de 2 GPUs, ya que hay ligeras diferencias en la ejecución de los 2 workers).

Puede apreciarse un ligero aumento del tiempo de ejecución en la versión distribuida, que se podría atribuir al aumento del número de las comunicaciones entre los workers al intercambiar datos. Sin embargo, en el tiempo final se refleja un notable decremento, concretamente se establece esta relación (**considerando el mismo tamaño del dataset y de los batches**):

$$ Tiempo DDP = \frac{Tiempo Secuencial}{Num Workers} $$

Es decir, la duracción del entrenamiento se reduce de forma **lineal** según el nº de workers disponibles, también se establece la siguiente relación:

$$ Steps Per Epoch = \frac{Filas Dataset}{Batch Size * Num Workers} $$

Siendo **NumWorkers = nº de nodos * nº de dispositivos (GPUs)**.

Acorde a esta relación, si se aumenta el nº de workers que intervienen en el entrenamiento, menos steps tendrá cada epoch (22500/(150*4) = 37,5 steps), por lo tanto menos tiempo durará el proceso. Si se aumenta el tamaño del dataset, más nº de steps habrá por epoch, por ende más durará el entrenamiento (por lo que se puede establecer un equilibrio entre aumentar el tamaño del dataset y el nº de los workers para mantener el tiempo de ejecución "estable"). El problema de DDP es que **no nos permite aumentar el tamaño del batch**, debido a que cada worker debe de albergar el **modelo entero + batch size en su GPU** (riesgo de un **CUDAOutOfMemory**).

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

En el caso de la captura, la gran mayoría del tiempo se invierte en la sincronización entre los workers, lo que indica una **ineficiencia** en las comunicaciones. Como ya se mencionó, el entrenamiento distribuido utiliza la estrategia **mirrored**, que usa funciones de comunicación colectivas para propagar los gradientes. Estas funciones establecen **puntos de sincronización** en cada step para obtener los gradientes y actualizar los pesos del modelo para así continuar con el resto de steps. Una mejora sería realizar estas sincronizaciones con **menos frecuencia para reducir el tiempo de las comunicaciones**, lo que por contraparte **afecta a la precisión del modelo**.

Por último, se muestra un panel con todos los detalles relevantes acerca de la ejecución de las **funciones de comunicación usadas por worker**. Se muestra el nº de llamadas a dichas funciones, el tamaño de los mensajes compartidos, la latencia asociada a la transmisión y el tiempo de transferencia de los datos.
- En la captura, se puede observar la ejecución de 2 funciones colectivas (broadcast y all_reduce) de la API **NCCL**, que es la implementación de NVIDIA de MPI. 

![image](https://github.com/user-attachments/assets/70576066-0bc8-4747-b4bb-0d56efd772ec)

Si volvemos a la vista de memoria que analizamos previamente en el BASELINE, cada worker consume la misma cantidad de memoria que en el entrenamiento secuencial con los mismos parámetros.

![image](https://github.com/user-attachments/assets/23516621-8095-42b1-9907-4c7b4dcd8737)


## Reassemble splited output files

Los archivos de salida de los entrenamientos ocupan en total en torno a unos **1,4 GB**:
- 3 salidas por pantalla del código de entrenamiento con 1, 2 y 4 GPUs (todas usan SGD como optimizador).
  - Al final muestran el tiempo de ejecución del entrenamiento y el valor de pérdida del test.
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
source ../../mypython/bin/activate
tensorboard --logdir=. --host `hostname -i` &
http://<IP_del_nodo>:6006/#pytorch_profiler
~~~

Puede iniciarse un nuevo entrenamiento del modelo desde el checkpoint guardado en el directorio **outputs**, sólo es necesario comentar y descomentar las respectivas líneas:

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/63ffcdafec2e0b958fc53723995908cec7c011d0/DISTRIBUTED/lightning_training.py#L116-L119
