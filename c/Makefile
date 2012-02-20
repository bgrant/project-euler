CSRC    = euler.c
CC      = clang
CFLAGS  = -gW -pedantic -O0

euler: $(CSRC)
	$(CC) $(CFLAGS) $(CSRC) -o euler

tags:
	rm -f tags ; ctags *.c *.h

clean:
	rm -f euler tags ; rm -rf *.dSYM

