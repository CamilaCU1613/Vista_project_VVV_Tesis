#!/bin/bash

# ###### Zona de ParÃ¡metros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=Training_1b278             #Nombre del job
#SBATCH -p medium                        #Cola a usar, Default=short (Ver colas y lÃ­mites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1                            #Nodos requeridos, Default=1
#SBATCH -n 1                            #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=24               #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=150000              #Memoria en Mb por CPU, Default=2048
#SBATCH --time=15-00:00:00                 #Tiempo mÃ¡ximo de corrida, Default=2 horas
#SBATCH --mail-user=c.cardenasu@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -o Training_1b278.o%j                 #Nombre de archivo de salida
#
########################################################################################

# ############################### Zona Carga de Modulos ################################
module load anaconda/python3.9
source activate Tesis
########################################################################################

# ###### Zona de Ejecucion de codigo y comandos a ejecutar secuencialmente #############
host=`/bin/hostname`
date=`/bin/hostname`
echo "Soy un JOB de prueba"
echo "Corri en la maquina: "$host
echo "Corri el: "$date

echo -e "Ejecutando Script de python \n"
python Training_sample_1_b278.py
echo -e "Finalice la ejecucion del script \n"
########################################################################################
