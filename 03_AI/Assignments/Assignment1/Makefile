# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Target executables
TARGETS = simple_reflex_agent model_based_agent goal_based_agent utility_based_agent

# Show all available targets when `make` is run
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all                - Compile all targets"
	@echo "  simple_reflex_agent - Compile Simple Reflex Agent"
	@echo "  model_based_agent   - Compile Model Based Agent"
	@echo "  goal_based_agent    - Compile Goal Based Agent"
	@echo "  utility_based_agent - Compile Utility Based Agent"
	@echo "  clean              - Remove all compiled files"

# Default target when running `make` without arguments
all: $(TARGETS)

# Compile individual targets
simple_reflex_agent: simple_reflex_agent.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

model_based_agent: model_based_agent.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

goal_based_agent: goal_based_agent.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

utility_based_agent: utility_based_agent.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

# Clean all generated files
clean:
	rm -f $(TARGETS) *.o
