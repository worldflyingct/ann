CC=gcc

ann:main.o
	$(CC) -static -o $@ $^

main.o:main.c
	$(CC) -c -o $@ $^

clean:
	del *.o
