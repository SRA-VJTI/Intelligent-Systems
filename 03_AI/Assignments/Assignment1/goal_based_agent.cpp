#include <iostream>
#include <vector>
#include <queue>
using namespace std;

bool isGoal(int x, int y, int goalX, int goalY) {
    return x == goalX && y == goalY;
}

void goalBasedAgent() {
    int rows, cols;
    cout << "Enter maze dimensions (rows cols): ";
    cin >> rows >> cols;

    vector<vector<int>> maze(rows, vector<int>(cols));
    cout << "Enter the maze layout (0 = empty, -1 = obstacle):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout<<"Enter value of row "<< i<< " and value "<<j<<" :"<<endl;
            cin >> maze[i][j];
        }
    }

    int startX, startY, goalX, goalY;
    cout << "Enter starting position (x y): ";
    cin >> startX >> startY;
    cout << "Enter goal position (x y): ";
    cin >> goalX >> goalY;

    int dx[] = {0, 1, 0, -1};
    int dy[] = {1, 0, -1, 0};

    queue<pair<int, int>> q;
    q.push({startX, startY});

    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    visited[startX][startY] = true;

    while (!q.empty()) {
        int x = q.front().first;
        int y = q.front().second;
        q.pop();

        cout << "Current Position: (" << x << ", " << y << ")\n";
        if (isGoal(x, y, goalX, goalY)) {
            cout << "Goal Reached!\n";
            return;
        }

        // Explore all possible moves
        for (int i = 0; i < 4; ++i) {
            int newX = x + dx[i];
            int newY = y + dy[i];

            if (newX >= 0 && newX < rows && newY >= 0 && newY < cols && maze[newX][newY] != -1 && !visited[newX][newY]) {
                q.push({newX, newY});
                visited[newX][newY] = true;
            }
        }
    }

    cout << "Goal cannot be reached!\n";
}

int main() {
    cout << "3. Goal-Based Agent:\n";
    goalBasedAgent();
    return 0;
}
