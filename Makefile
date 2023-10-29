CC	= gcc
CFLAGS	= -O0 -g -Wall -I.
LDFLAGS = -L. -lm -lpthread -lnbody -lcheck -lsubunit -fopenmp
INTERNAL_LIBS = libnbody.a
VERBOSE	=

TARGET	= nbody_brute_force nbody_barnes_hut
TESTS_TARGET = nbody_brute_force_test nbody_barnes_hut_test

NBODY_OBJECTS = utils/nbody/nbody_tools.o utils/nbody/nbody_alloc.o
TEST_BRUTE_FORCE_OBJECTS = tests/nbody_brute_force_test.o tests/valid_impl/nbody_brute_force.o sequential/nbody_brute_force.o
TEST_BARNES_HUT_OBJECTS = tests/nbody_barnes_hut_test.o tests/valid_impl/nbody_barnes_hut.o sequential/nbody_barnes_hut.o

DEFINE =

ENABLE_UI = "true"

ifeq ($(ENABLE_UI), "true")

UI_OBJECTS = utils/ui/ui.o utils/ui/xstuff.o
DEFINE += -DDISPLAY
LDFLAGS += -lui -lX11
INTERNAL_LIBS += libui.a

libui.a: $(UI_OBJECTS)
	ar rcs libui.a $(UI_OBJECTS)

endif

# DEFINE += -DDUMP_RESULT

all: $(TARGET) tests

tests: $(TESTS_TARGET)

nbody_brute_force_test: $(TEST_BRUTE_FORCE_OBJECTS) $(INTERNAL_LIBS)
	$(CC) $(VERBOSE) -o $@ $(TEST_BRUTE_FORCE_OBJECTS) $(LDFLAGS)

nbody_barnes_hut_test: $(TEST_BARNES_HUT_OBJECTS) $(INTERNAL_LIBS)
	$(CC) $(VERBOSE) -o $@ $(TEST_BARNES_HUT_OBJECTS) $(LDFLAGS)

nbody_brute_force: sequential/nbody_brute_force.o main.o $(INTERNAL_LIBS)
	$(CC) $(VERBOSE) -o $@ main.o $< $(LDFLAGS)

nbody_barnes_hut: sequential/nbody_barnes_hut.o main.o $(INTERNAL_LIBS)
	$(CC) $(VERBOSE) -o $@ main.o $< $(LDFLAGS)

libnbody.a: $(NBODY_OBJECTS)
	ar rcs libnbody.a $(NBODY_OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< $(VERBOSE) $(DEFINE)

clean:
	rm -f --verbose *.o *.a $(TARGET) $(TESTS_TARGET)
	find . -type f -name "*.o" -print -delete

launch_tests:
	./nbody_brute_force_test
	./nbody_barnes_hut_test
