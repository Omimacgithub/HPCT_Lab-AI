# HPCT - Lab-AI / Fine-tuning BERT model for question answering with SQUAD dataset
Git repository for AI lab

## Colaborators
- Omar Montenegro Macía
- Álvaro Pardo Fente

## Table of contents

- [SQUAD](#squad)
- [BERT](#bert)
- [Explanation of the code](#explanation-of-the-code)
- [How to run it?](#how-to-run-it)
  - [Check training profile data](#check-training-profile-data)
- [Profiling outputs](#profiling-outputs)
	- [Execution times](#execution-times)
 	- [Tensorboard](#tensorboard)
- [Reassemble splited output files](#reassemble-splited-output-files)
## SQUAD
Este dataset nos proporciona un json con los siguientes campos significativos:

- context: proporciona información detallada acerca de un tema (ejemplo: bibliografía de Fréderic Chopin)
- question: una pregunta acerca de un tema
- answer: diccionario con los siguientes elementos:
	- text: respuesta a la pregunta (información para el dataset de test)
	- answer\_start: offset en nº de caracteres donde comienza la respuesta incluida en el context (información para el dataset de test).

## BERT
El objetivo es entrenar al modelo BERT para que dado un contexto **sepa responder correctamente** a las preguntas que se le hacen del tema relacionado con el contexto.

Para este modelo son necesarias las siguientes entradas(features):

- input\_ids: en esta entrada se introduce el context y las preguntas asociadas. Cada valor representa un índice que corresponde a la posición de una palabra en un wordlist que usa el modelo(proceso de tokenización de los datos).
- attention\_mask: indica si el token es una máscara (0) o no (1).

Adicionalmente, crearemos 2 parámetros que el modelo usará a modo de salidas esperadas para calcular el valor de pérdida (loss):

- start\_position: igual al campo **answer_start** del dataset.
- end\_position: offset en nº de caracteres donde **termina** la respuesta incluida en el context.

El modelo computa su respectivo start\_position (start\_logits) y end\_position (start\_logits) de la que cree que es la respuesta correcta y los compara con las salidas esperadas (calcula la pérdida).

## Explanation of the code

El entrenamiento se divide en 2 ficheros:

- tokenize\_squad.py: fichero que descarga el dataset y lo tokeniza para usarlo en el modelo de BERT.

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/tokenize_squad.py#L91

-
	- Transforma (tokeniza) los campos "question" y "context" en input\_ids válidos para el modelo de BERT (dado que la información del context puede ser demasiado extensa y el tamaño de los input\_ids de BERT es limitado, es necesario fragmentarla en varios inputs\_ids).

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/tokenize_squad.py#L12-L23

-
	- Computa los campos start\_position y end\_position que serán usados por BERT para calcular el valor de loss.


-
	- Devuelve un objeto datasets.arrow\_dataset.Dataset con todas las entradas necesarias para el modelo y lo guarda en un fichero.

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/tokenize_squad.py#L93-L97

- lightning_training.py: crea la clase que ejecuta el entrenamiento del modelo de BERT y genera información de profiling del mismo.
	- Se define la clase como un LightningModule con el modelo. Debe de implementar los siguientes métodos:
 		- forward: invoca al modelo BERT con los parámetros necesarios.

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/lightning_training.py#L35-L38

-
  		- training\_step: Devuelve el valor de pérdida generado en el step de entrenamiento. Llama a la función forward, que devuelve las salidas generadas por el modelo (start\_logits y end\_logits), de las que se calcula el valor de pérdida respecto a las salidas esperadas (start\_positions y end\_positions). Al usar pytorch\_lightning no es necesario declarar métodos como optimizer.step() o optimizer.zero_grad() en el entrenamiento.
 
https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/lightning_training.py#L40-L54

-
		- validation_step y test_step en caso de que se quiera realizar estas fases, ejecutan el mismo código que la fase de training.
  		- configure_optimizers: devuelve los optimizadores que se usarán en el entrenamiento (sólo SGD) y los schedulers que modifican el valor del learning rate (en este caso ninguno).

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/lightning_training.py#L87C9-L89

-
	- Carga el Dataset creado del anterior archivo .py (o genera uno si no está creado).

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/lightning_training.py#L95-L103

-
	- Carga el Dataset en un objeto DataLoader para poder iterarlo en batches.

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/lightning_training.py#L111-L121

-
	- Crea un objeto PyTorchProfiler con los argumentos necesarios:
		- **schedule:** decide que acción del profiler ejecutar en cada paso (step)
			- wait: el profiler se deshabilita
			- warmup: el profiler graba información, pero los resultados son descartados
			- active: el profiler graba información y guarda los resultados
   			- repeat: la secuencia formada por los 3 parámetros anteriores se repite x veces.
                - **on_trace_ready:** es un hook que se activa cuando el scheduler devuelve ProfilerAction.RECORD_AND_SAVE (que es cuando se acaban todos los pasos del profiler wait+warmup+active). En este caso la acción que ejecuta es la de guardar los resultados en el directorio declarado.

https://github.com/Omimacgithub/HPCT_Lab-AI/blob/c8d5ed2a8e86acc0589b1808e0482d7645e845cf/lightning_training.py#L130-L140

-
	- Ejecuta el bucle de entrenamiento al mismo tiempo que guarda la información de profiling.



## How to run it?

**NOTA:** la visualización de la información de profiling en tensorboard **no está soportada para el navegador Safari**.

Para crear el venv (se crea en el directorio del repositorio, si ya se encuentra creado se omite este paso) y lanzar el trabajo para entrenar el modelo ejecutamos el script **launch.sh**:

~~~shell
./launch.sh
~~~

La ejecución devuelve un fichero .out que contiene la duración en minutos del entrenamiento y un directorio **runs/BERTSQUAD** con la información de profiling.

### Check training profile data

**En el directorio del repositorio** ejecutamos tensorboard con los datos generados:

~~~shell
source ./mypython/bin/activate
tensorboard --logdir=./runs --host `hostname -i` &
~~~

Accedemos desde el navegador web a la sección de profiling en tensorboard:

~~~shell
http://<IP_del_nodo>:6006/#pytorch_profiler
~~~

Si queremos salir del venv:

~~~shell
deactivate
~~~

## Profiling outputs

### Execution times

Se recogen 2 salidas del entrenamiento de BERT, 1 salida usando la optimización AdamW y la otra usando SGD. Se usó la siguiente configuración.

-  Fase de entrenamiento con 22500 filas del dataset de entrenamiento.
	- Tamaño de batch de 150 (22500/150 = 150 steps)
-  Fase de validación con las 200 primeras filas del dataset de validación y test con las 200 últimas filas del dataset de validación.
	- Tamaño de batch de 8 (200/8 = 25 steps)
 - 6 epochs.

| Optimizers | Step 0      | Step 1      | Step 2      | Step 3      | Step 4      | Step 5      | Avg step     | Avg epoch (150 steps)   | Total exc    |
| ---------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| AdamW      | 2607,112 ms | 2115,140 ms | 2106,955 ms | 2107,439 ms | 2109,853 ms | 2110,829 ms | 2192,888 ms | 5,48222 mins | 33,2294 mins |
| SGD        | 2505,104 ms  | 2145,509 ms | 2135,011 ms | 2138,431 ms | 2140,558 ms | 2142,426 ms | 2201,173 ms | 5,5029325 mins | 33,0592 mins |

La duración del step 0 destaca respecto de las demás, ya que al principio del entrenamiento los primeros steps son de [calentamiento](https://medium.com/@MarkAiCode/mastering-pytorch-inference-time-measurement-22da0eaebab7) y llevan más tiempo de lo normal debido a factores como el **cache warming** o el **JIT compilation**.

### Tensorboard

Al entrar en tensorboard podemos ver un resumen del entrenamiento. Se detalla información como la GPU utiliza (en este caso una NVIDIA A100) y su uso en % (90.32 en este caso), la duración de cada step desglosada en varias categorías (vemos como la ejecución de los kernels en la GPU fué lo más costoso con diferencia). 

- La métrica de average time step no muestra el tiempo promedio **real** de cada step del entrenamiento, ya que tiene en cuenta un paso adicional en el que se invoca al método **cudaDeviceSynchronize** (bloquea la CPU hasta que todas las operaciones de la GPU hayan terminado), que tiende a bajar mucho la media.

Adicionalmente la herramienta otorga recomendaciones para mejorar el rendimiento del entrenamiento (como puede verse en la parte inferior para este entrenamiento no hay ninguna recomendación disponible).

![image](https://github.com/user-attachments/assets/24c5e4dc-d062-49e6-a139-718b72025c8b)

El apartado GPU kernel muestra el tiempo de ejecución de cada kernel en la GPU y además la ejecución de kernels en tensor cores si se diese el caso (en este entrenamiento no aplica).

![image](https://github.com/user-attachments/assets/f6101236-d30c-48f0-9a88-53b94b932acb)


El apartado Trace devuelve información relacionada con el instante de ejecución de cada función del código del entrenamiento. Gracias a esta información podemos comprobar que el entrenamiento se ejecuta adecuadamente (se invoca al optimizador), el tiempo de cada step (Wall Duration de la ventana inferior que se muestra en la imagen), entre otra información relevante.

Si posicionamos el ratón en un bloque de funciones, pulsando w se pueden ampliar los detalles de la misma y se pueden reducir pulsando s.

![TRACE](https://github.com/user-attachments/assets/d1cc9222-a391-4987-844c-e75e039ca8a8)

Si navegamos entre los spans, podemos ver información de las fases de validación y de test.

![EVAL](https://github.com/user-attachments/assets/858eee61-3c9f-4700-ab9c-4aade3e9b541)

![TEST](https://github.com/user-attachments/assets/5f2400a7-1001-4744-bda7-72a51d3a2eea)

Las capturas anteriores muestran la ejecución de las funciones en **un thread de la CPU**, si nos desplazamos hacia abajo en los datos podemos ver las ejecuciones de los threads asociados a las GPUs. Para las GPUs se muestran 2 apartados especiales, que son el valor de uso de la GPU en cada instante de tiempo y la eficiencia.

![GPU](https://github.com/user-attachments/assets/d8e798ee-2366-484c-910f-a09634097c04)

Si seleccionamos la opción "Flow Events" podemos ver a través de lineas de color verde la relación entre las operaciones realizadas por los threads de la CPU con la ejecución de los respectivos kernels en los threads de la GPU.

![image](https://github.com/user-attachments/assets/b5ad5c77-681e-44db-808a-7a4af7176899)

Otra pestaña con información útil es la de memoria. En ella podemos ver la ocupación de la memoria por dispositivo a lo largo del tiempo. La captura inferior nos muestra el uso de memoria de la GPU por instante de tiempo, que llega hasta un máximo de 33 GB de memoria ocupada, lo que indica que la GPU casi se queda sin espacio libre en el entrenamiento (por ejemplo al aumentar el tamaño de los batches, es importante tener en cuenta este dato para no provocar un error CUDA.OutOfMemory). El trazado azul nos informa de que se han producido picos de uso de memoria al principio de la ejecución de cada step (cada 2000 ms aproximadamente) y que en el resto del tiempo la memoria se infrautiliza.

Debajo de la gráfica se muestra un listado con las funciones que han reservado memoria indicando la cantidad en KB y la duración de la reserva.

![image](https://github.com/user-attachments/assets/42370bc7-2e51-4c8e-ab20-aab1e3fa73b7)


## Reassemble splited output files

Los archivos de salida de los entrenamientos ocupan en total en torno a unos **900 MB**:
- 2 salidas por pantalla del código de entrenamiento, 1 con AdamW y otra con SGD
- Archivos de profiling de los 2 entrenamientos para examinar con tensorboard
- 1 checkpoint del entrenamiento final del modelo usando SGD (directorio version\_2).

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
