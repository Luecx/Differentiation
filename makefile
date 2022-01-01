CC       = g++
SRC      = $(wildcard *.cpp */*.cpp */*/*.cpp */*/*/*.cpp)
LIBS     = -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive
FOLDER   = bin/
ROOT     = ./
NAME     = Differentiation
EXE      = $(ROOT)$(FOLDER)$(NAME)_$(MAJOR).$(MINOR)
MINOR    = 1
MAJOR    = 0
ifeq ($(OS),Windows_NT)
    PREFIX := windows
    SUFFIX := .dll
else
    PREFIX := linux
    SUFFIX := .dll
endif

WFLAGS = -std=c++17 -Wall -Wextra -Wshadow
CFLAGS = -O3 $(WFLAGS) -DNDEBUG
PFLAGS = -O0 $(WFLAGS) -DNDEBUG -p -pg
DFLAGS = -O0 $(WFLAGS) -g

SSEFLAGS    = -msse
SSE2FLAGS   = $(SSEFLAGS) -msse2
SSE3FLAGS   = $(SSE2FLAGS) -msse3
SSE41FLAGS  = $(SSE3FLAGS) -msse4.1
SSE42FLAGS  = $(SSE41FLAGS) -msse4.2
POPFLAGS    = $(SSE42FLAGS) -DUSE_POPCNT -mpopcnt
AVXFLAGS    = $(POPFLAGS) -mavx
AVX2FLAGS   = $(AVXFLAGS) -mavx2
AVX512FLAGS = $(AVX2FLAGS) -mavx512f -mavx512bw -mavx512dq

OPENMPFLAGS = -fopenmp

MAKROS      = -DMINOR_VERSION=$(MINOR) -DMAJOR_VERSION=$(MAJOR)

NATIVEFLAGS = -march=native -flto
STATICFLAGS = -static -static-libgcc -static-libstdc++
DLLFLAGS    = -shared -fPIC

native:
	mkdir -p $(ROOT)$(FOLDER)
	$(CC) $(CFLAGS) $(SRC) $(NATIVEFLAGS) $(OPENMPFLAGS) $(MAKROS) $(LIBS) -o $(EXE)-x64-$(PREFIX)-native$(SUFFIX)

shared:
	mkdir -p $(ROOT)$(FOLDER)
	$(CC) $(CFLAGS) $(SRC) $(NATIVEFLAGS) $(OPENMPFLAGS) $(DLLFLAGS) $(MAKROS) $(LIBS)
# -o $(EXE)-x64-$(PREFIX)-shared$(SUFFIX)

