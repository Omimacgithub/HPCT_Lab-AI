# HPCT - Lab-AI / Fine-tuning BERT model for question answering with SQUAD dataset
Git repository for AI lab

## Colaborators
- Omar Montenegro Macía
- Álvaro Pardo Fente

## SQUAD
Este dataset nos proporciona un json con los siguientes campos significativos:

- context: proporciona información detallada acerca de un tema (ejemplo: bibliografía de Fréderic Chopin)
- question: una pregunta acerca de un tema
- answer: diccionario con los siguientes elementos:
	- text: respuesta a la pregunta (información para el dataset de test)
	- answer\_start: offset en nº de caracteres donde comienza la respuesta incluida en el context (información para el dataset de test).

El objetivo es entrenar al modelo BERT para que dado un context **sepa responder correctamente** a las preguntas que se le hacen del tema relacionado con el contexto.
##BERT

Para este modelo son necesarias las siguientes entradas(features):

- input\_ids: en esta entrada se introduce el context y las preguntas asociadas. Cada valor representa un índice que corresponde a la posición de una palabra en un wordlist que usa el modelo(proceso de tokenización de los datos).
- attention\_mask: indica si el token es una máscara (0) o no (1).

Adicionalmente, añadimos 2 entradas más que el modelo usará para calcular el valor de pérdida (loss):

- start\_position: igual al campo **answer_start** del dataset
- end\_position: offset en nº de caracteres donde **termina** la respuesta incluida en el context

El modelo computa su respectivo start\_position y end\_position de la que cree que es la respuesta correcta y los compara con los 2 valores que se pasan como entradas (calcula la pérdida).

## Explanation of the code

El entrenamiento se divide en 2 ficheros:

- tokenize_squad.py: fichero que descarga el dataset y lo tokeniza para usarlo en el modelo de BERT.
- training.py: ejecuta el entrenamiento con el modelo de BERT y genera información de profiling del mismo.

## How to run it?

**NOTA:** la visualización de la información de profiling en tensorboard **no está soportada para el navegador Safari**.

Para crear el venv y lanzar el trabajo para entrenar el modelo ejecutamos el script **createnv.sh**:

~~~shell
./createnv.sh
~~~

La ejecución devuelve un fichero .out que contiene la duración en minutos del entrenamiento y un directorio **runs/BERTSQUAD** con la información de profiling.

### Check training profile data

**En el directorio del repositorio** ejecutamos tensorboard con los datos generados:

~~~shell
tensorboard --logdir=./runs --bind_all &
~~~

Accedemos desde el navegador web a la sección de profiling en tensorboard:

~~~shell
http://<IP_del_nodo>:6006/#pytorch_profiler
~~~
