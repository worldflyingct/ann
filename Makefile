CC=gcc

ann:main.o dataset.o nn.o
	$(CC) -static -o $@ $^

main.o:main.c dataset.h nn.h config.h
	$(CC) -c -o $@ main.c

dataset.o:dataset.c dataset.h config.h
	$(CC) -c -o $@ dataset.c

nn.o:nn.c nn.h config.h
	$(CC) -c -o $@ nn.c

clean:
	del *.o
