CSRC    = euler.c
CFLAGS  = -gW -pedantic -O0

euler: $(CSRC)
	clang $(CFLAGS) $(CSRC) -o euler

tags:
	rm tags ; ctags *.c *.h

clean:
	rm -f euler tags ; rm -rf *.dSYM

