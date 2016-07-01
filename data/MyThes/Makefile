
CXX=g++

#CXXFLAGS= -O0 -g -Wall -ansi -pedantic -I.
#LDFLAGS=-L. -lmythes -ldmalloc

CXXFLAGS= -O2 -Wall -ansi -pedantic -I.
LDFLAGS=-L. -lmythes

LIBS=libmythes.a

AR=ar rc
RANLIB=ranlib

OBJS = mythes.o 

all: lqrz_lookup

libmythes.a: $(OBJS)
	$(AR) $@ $(OBJS)
	-@ ($(RANLIB) $@ || true) >/dev/null 2>&1

lqrz_lookup: lqrz_lookup.o $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ lqrz_lookup.o $(LDFLAGS)

%.o: %.cxx 
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o *~ lqrz_lookup libthes.a

distclean:	clean

depend:
	makedepend -- $(CXXFLAGS) -- *.[ch]xx

# DO NOT DELETE THIS LINE -- make depend depends on it.

mythes.o: mythes.hxx
lqrz_lookup.o: mythes.hxx 

