# for para el tamano del kernel
for kernel in 3 7 11 15
do
	# for para el numero de hilos con respecto al kernel
	for NumHilos in 32 128 1024 4096 8192 32768
	do


		printf "Tiempo para imagen de 720p con un kernel de $kernel y con # $NumHilos hilos \n"
		time ./blur_cuda 720.jpg $kernel $NumHilos


		printf "Tiempo para imagen de 1080p con un kernel de $kernel y con # $NumHilos hilos \n"
		time ./blur_cuda 1080.jpg $kernel $NumHilos


		printf "Tiempo para imagen de 4k con un kernel de $kernel y con # $NumHilos hilos \n"
		time ./blur_cuda 4k.jpg $kernel $NumHilos


	done
done
