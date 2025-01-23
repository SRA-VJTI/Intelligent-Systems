# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Target executables
TARGETS = main

# Show all available targets when `make` is run
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  main  - Compile main.py"
	@echo "  clean - Remove all compiled files"


# Compile individual targets
main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

# Clean all generated files
clean:
	rm -f $(TARGETS) *.o
