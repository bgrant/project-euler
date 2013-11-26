SRC77	= euler.f
SRC90	= euler.f90
CC		= gfortran
FLAGS	= -fimplicit-none

all: euler77 euler90

python77:
	f2py -c $(SRC77) -m euler77

python90:
	f2py -c $(SRC90) -m euler90

euler77: $(SRC77)
	$(CC) $(FLAGS) $(SRC77) -o euler77

euler90: $(SRC90)
	$(CC) $(FLAGS) $(SRC90) -o euler90

clean:
	rm -f euler77 euler90 *.so
