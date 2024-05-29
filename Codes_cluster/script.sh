#!/bin/bash

# ###### Zona de ParÃ¡metros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=VVV_cluster             #Nombre del job
#SBATCH -p medium                        #Cola a usar, Default=short (Ver colas y lÃ­mites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1                            #Nodos requeridos, Default=1
#SBATCH -n 1                            #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=5               #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=60000              #Memoria en Mb por CPU, Default=2048
#SBATCH --time=1:00:00:00                 #Tiempo mÃ¡ximo de corrida, Default=2 horas
#SBATCH --mail-user=c.cardenasu@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -o VVV_cluster.o%j                 #Nombre de archivo de salida
#
########################################################################################

# ############################### Zona Carga de Modulos ################################
module load anaconda
source activate Spectro
########################################################################################

# ###### Zona de Ejecucion de codigo y comandos a ejecutar secuencialmente #############
host=`/bin/hostname`
date=`/bin/hostname`
echo "Soy un JOB de prueba"
echo "Corri en la maquina: "$host
echo "Corri el: "$date

echo -e "Ejecutando Script de python \n"
python archivogg.py
echo -e "Finalice la ejecucion del script \n"
########################################################################################
