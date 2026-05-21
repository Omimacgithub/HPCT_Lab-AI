# HPCT - Lab-AI / Profiling overview of BERT fine-tuning for question answering with SQUAD dataset
Git repository for AI lab

## Colaborators
- Omar Montenegro Macía
- Álvaro Pardo Fente

## Objectives

Esta práctica se divide en 3 grandes objetivos:

  - El primer objetivo es obtener un modelo (BASELINE) resultado de realizar un **fine-tuning** del modelo de BERT para responder preguntas del dataset SQUAD por medio de una GPU **NVIDIA A100 del Finisterrae III**.
  - El segundo objetivo es obtener una nueva versión del modelo (DISTRIBUTED) siguiendo una estrategia de **entrenamiento distribuido** utilizando **dos nodos del Finisterrae III, cada uno con dos GPUs NVIDIA A100**.
  - El último objetivo es comparar la ejecución de ambos entrenamientos, resaltando el menor tiempo de entrenamiento de la versión DISTRIBUTED respecto de la versión BASELINE (ver README del directorio DISTRIBUTED).

## Go to

- [BASELINE](BASELINE/)
- [DISTRIBUTED](DISTRIBUTED/)
