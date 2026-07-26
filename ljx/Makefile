# LJX build — clang is required for the musttail interpreter kernel.
CXX := clang++-18
# -MMD -MP: regenerate objects when a header changes. Without this a layout
# change silently leaves stale .o files linked against the old offsets.
CXXFLAGS := -std=c++20 -O2 -g -Wall -Wextra -Iinclude -MMD -MP

SRC := $(wildcard src/core/*.cpp) $(wildcard src/rt/*.cpp) $(wildcard src/gc/*.cpp) \
       $(wildcard src/vm/*.cpp) $(wildcard src/fe/*.cpp) $(wildcard src/jit/*.cpp)
OBJ := $(patsubst src/%.cpp,build/%.o,$(SRC))
DEP := $(OBJ:.o=.d) build/main.d build/test.d

all: ljx ljx_test

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

ljx: $(OBJ) build/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@

build/main.o: src/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

ljx_test: $(OBJ) build/test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

build/test.o: tests/test_main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

check: ljx_test ljx
	./ljx_test
	./tests/run_lua_tests.sh

clean:
	rm -rf build ljx ljx_test

-include $(DEP)

.PHONY: all check clean
