# for para el tamano del kernel
echo "----------------------------------------------------------" >> resultados.txt
for kernel in 3 7 11 15
do
	# for para el numero de hilos con respecto al kernel
	for NumHilos in 32 128 1024 4096 8192 32768
	do


		echo "Tiempo para imagen de 720p con un kernel de $kernel y con # $NumHilos hilos \n" >> resultados.txt
		(time ./blur_cuda 720.jpg $kernel $NumHilos) &>> resultados.txt


		echo "Tiempo para imagen de 1080p con un kernel de $kernel y con # $NumHilos hilos \n">> resultados.txt
		(time ./blur_cuda 1080.jpg $kernel $NumHilos) &>> resultados.txt


		echo "Tiempo para imagen de 4k con un kernel de $kernel y con # $NumHilos hilos \n">> resultados.txt
		(time ./blur_cuda 4k.jpg $kernel $NumHilos) &>> resultados.txt


	done
done
