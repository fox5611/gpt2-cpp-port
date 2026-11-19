CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -Wall -Wextra -pthread

HEADERS := $(wildcard *.h)
SRCS    := $(wildcard *.cpp)
BINS    := $(basename $(SRCS))

.PHONY: all clean
all: $(BINS)

%: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
