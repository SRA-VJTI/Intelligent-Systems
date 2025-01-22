#include <iostream>
#include <vector>
using namespace std;

void modelBasedAgent() {
    int n;
    cout << "Enter the number of rooms: ";
    cin >> n;
    vector<int> roomStates(n);
    cout << "Enter the state of each room (0 = Clean, 1 = Dirty):\n";
    for (int i = 0; i < n; ++i) {
        cout << "Room " << i + 1 << ": ";
        cin >> roomStates[i];
    }

    for (size_t i = 0; i < roomStates.size(); ++i) {
        if (roomStates[i] == 1) {
            cout << "Room " << i + 1 << " is Dirty. Cleaning...\n";
            roomStates[i] = 0; // Update the state to Clean
        } else {
            cout << "Room " << i + 1 << " is already Clean.\n";
        }
    }
}

int main() {
    cout << "2. Model-Based Agent:\n";
    modelBasedAgent();
    return 0;
}
