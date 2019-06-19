all:	
	nvcc  -Wno-deprecated-gpu-targets  `pkg-config --cflags --libs opencv`   -I /usr/local/cuda-7.5/include/ -L /usr/local/cuda-7.5/lib64/ -lcuda -lopencv_core -lopencv_highgui -lopencv_imgproc $(wildcard *.cu) -o borrosoCUDA
