#include <iostream>
#include <vector>
#include <cmath>
#include <climits>
using namespace std;

void utilityBasedAgent() {
    int n;
    cout << "Enter the number of packages: ";
    cin >> n;
    vector<pair<int, int>> packages(n);

    cout << "Enter the coordinates of each package (x y):\n";
    for (int i = 0; i < n; ++i) {
        cout << "Package " << i + 1 << ": ";
        cin >> packages[i].first >> packages[i].second;
    }

    pair<int, int> dronePosition = {0, 0};
    double shortestDistance = INT_MAX;
    pair<int, int> closestPackage;

    for (auto package : packages) {
        double distance = sqrt(pow(package.first - dronePosition.first, 2) + pow(package.second - dronePosition.second, 2)); // Euclidean distance between two coordinates
        if (distance < shortestDistance) {
            shortestDistance = distance;
            closestPackage = package;
        }
    }
    cout << "Drone delivers to closest package at position (" << closestPackage.first << ", " << closestPackage.second << ")\n";
}

int main() {
    cout << "4. Utility-Based Agent:\n";
    utilityBasedAgent();
    return 0;
}
