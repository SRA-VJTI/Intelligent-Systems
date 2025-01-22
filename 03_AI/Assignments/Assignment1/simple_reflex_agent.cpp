#include <iostream>
using namespace std;

void simpleReflexAgent() {
    int currentTemp;
    cout << "Enter the current temperature: ";
    cin >> currentTemp;
    int threshold = 22; // Threshold temperature
    if (currentTemp > threshold) {
        cout << "Thermostat Action: Turn OFF the heater\n";
    } else {
        cout << "Thermostat Action: Turn ON the heater\n";
    }
}

int main() {
    cout << "1. Simple Reflex Agent:\n";
    simpleReflexAgent();
    return 0;
}
