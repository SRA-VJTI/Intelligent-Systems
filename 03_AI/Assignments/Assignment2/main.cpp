#include<bits/stdc++.h>
#include <chrono>
using namespace std;
const int n = 3;
int dfsmoves = 0;

// Print the current configuration of grid 
void print(vector<vector<int>>&grid) {
    for(auto &i: grid) {
        for(auto &j: i) {
            cout<<j<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

// Generate a random initial grid for the 8-puzzle problem
vector<vector<int>> gen() {
    vector<vector<int>>grid(n, vector<int>(n, 0));
    vector<int>nums = {1, 2, n, 4, 5, 6, 7, 8, 0};
    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) {
            random_shuffle(nums.begin(), nums.end());
            grid[i][j] = nums[nums.size()-1];
            nums.pop_back();
        }
    }
    return grid;
}

// Manhattan distance heuristic: calculates total distance of tiles from their goal positions
int manhat(int x1, int x2, int y1, int y2) {
    return abs(x1-x2) + abs(y1-y2);
}

// Heuristic 2: Manhattan distance-based evaluation
int h2(vector<vector<int>>& start, vector<vector<int>>& goal) {
    unordered_map<int, pair<int, int>> mp;
    int final = 0;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            if (goal[i][j] != 0) {
                mp[goal[i][j]] = {i, j};
            }
        }
    }

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            if (start[i][j] != 0) {
                int x2 = mp[start[i][j]].first;
                int y2 = mp[start[i][j]].second;
                final += manhat(i, x2, j, y2);
            }
        }
    }

    return final;
}


//Calculate heuristics by misplaced tiles
int h1(vector<vector<int>>& start, vector<vector<int>>& goal) {
    int ans = 0;
    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) {
            if(goal[i][j] == start[i][j]) continue;
            ans++;
        }
    }
    return ans;
}

// Check if puzzle configuration is solvable by counting inversions
bool solvable(vector<vector<int>>&start, vector<vector<int>>&goal) {
    vector<int>endflat;
    vector<int>startflat;
    for(int i = 0; i<n; i++) {
        for(int j = 0; j<n; j++) {
            if(start[i][j])
                startflat.push_back(start[i][j]);
            if(goal[i][j])
                endflat.push_back(goal[i][j]);
        }
    }
    unordered_map<int,int>mp;
    for(int i = 0; i<8; i++) {
        mp[endflat[i]] = i;;
    }
    for(int i = 0; i<8; i++) {
        startflat[i] = mp[startflat[i]];
    }
    int count = 0;
    int n = startflat.size();
    for(int i = 0; i<n; i++) {
        int cur = startflat[i];
        for(int j = i+1; j<n; j++) {
            int curj = startflat[j];
            if(cur>curj) count++;
        }
    }
    if(count&1) {
        return false;
    } else return true;
}

bool isvalid(int i, int j) {
    return !(i<0 || j<0 || i>=3 || j>=3);
}

struct state {
    vector<vector<int>>grid;
    int x, y, moves;
};

// dfs algo
bool dfs(vector<vector<int>>& start, vector<vector<int>>& goal, int &moves) {
    stack<state>st;
    set<vector<vector<int>>>vis;
    for(int i = 0; i<3; i++) {
        bool flag = 0;
        for(int j = 0; j<3; j++) {
            if(!start[i][j]) {
                st.push({start, i, j, 0});
                flag = 1;
                break;
            }
        }
        if(flag) break;
    }

    while(!st.empty()) {
        auto curstate = st.top();
        st.pop();
        auto cur = curstate.grid;
        vis.insert(cur);
        int i = curstate.x;
        int j = curstate.y;
        int curmoves = curstate.moves;
        if(cur == goal) {
            moves = curmoves;
            return 1;
        }
        int dx[4] = {0, 0, 1, -1};
        int dy[4] = {1, -1, 0, 0};
        for(int idx = 0; idx<4; idx++) {
            int x = i + dx[idx];
            int y = j + dy[idx];
            if(isvalid(x, y)) {
                swap(cur[i][j], cur[x][y]);
                if(vis.count(cur) == 0)
                    st.push({cur, x, y, curmoves+1});
                swap(cur[i][j], cur[x][y]);
            }
        }
    }
    return 0;
}

//bfs algo
bool bfs(vector<vector<int>>& start, vector<vector<int>>& goal, int i, int j, int &moves) {
    set<vector<vector<int>>> vis;
    queue<pair<vector<vector<int>>, pair<int, int>>> q;

    q.push({start, {i, j}});
    vis.insert(start);

    while (!q.empty()) {
        auto current = q.front();
        q.pop();
        auto cur = current.first;
        int i = current.second.first;
        int j = current.second.second;

        if (cur == goal) {
            return true;
        }

        int dx[4] = {0, 0, 1, -1};
        int dy[4] = {1, -1, 0, 0};

        for (int idx = 0; idx < 4; idx++) {
            int x = i + dx[idx];
            int y = j + dy[idx];

            if (isvalid(x, y)) {
                swap(cur[i][j], cur[x][y]);

                if (!vis.count(cur)) {
                    vis.insert(cur);
                    q.push({cur, {x, y}});
                }

                swap(cur[i][j], cur[x][y]);
            }
        }

        moves++;
    }

    return false;
}

// Best FS algo using h1 
bool bestfsh1(vector<vector<int>>& start, vector<vector<int>>& goal, int i, int j, int &moves) {
    set<vector<vector<int>>> vis;
    priority_queue<pair<int, pair<vector<vector<int>>, pair<int, int>>>, vector<pair<int, pair<vector<vector<int>>, pair<int, int>>>>, greater<pair<int, pair<vector<vector<int>>, pair<int, int>>>>> q;
    int temp = h1(start, goal);
    q.push({temp, {start, {i, j}}});
    vis.insert(start);

    while (!q.empty()) {
        auto current = q.top();
        q.pop();
        auto cur = current.second.first;
        int i = current.second.second.first;
        int j = current.second.second.second;

        if (cur == goal) {
            return true;
        }

        int dx[4] = {0, 0, 1, -1};
        int dy[4] = {1, -1, 0, 0};
        
        for (int idx = 0; idx < 4; idx++) {
            int x = i + dx[idx];
            int y = j + dy[idx];
            if(!isvalid(x, y)) continue;
            swap(cur[i][j], cur[x][y]);
            if (!vis.count(cur)) {
                int h = h1(cur, goal);
                vis.insert(cur);
                q.push({h, {cur, {x, y}}});
            }

            swap(cur[i][j], cur[x][y]);
        }

        moves++;
    }
    return false;
}

// Best FS algo using h2
bool bestfsh2(vector<vector<int>>& start, vector<vector<int>>& goal, int i, int j, int &moves) {
    set<vector<vector<int>>> vis;
    priority_queue<pair<int, pair<vector<vector<int>>, pair<int, int>>>, vector<pair<int, pair<vector<vector<int>>, pair<int, int>>>>, greater<pair<int, pair<vector<vector<int>>, pair<int, int>>>>> q;
    int temp = h2(start, goal);
    q.push({temp, {start, {i, j}}});
    vis.insert(start);

    while (!q.empty()) {
        auto current = q.top();
        q.pop();
        auto cur = current.second.first;
        int i = current.second.second.first;
        int j = current.second.second.second;

        if (cur == goal) {
            return true;
        }

        int dx[4] = {0, 0, 1, -1};
        int dy[4] = {1, -1, 0, 0};
        
        for (int idx = 0; idx < 4; idx++) {
            int x = i + dx[idx];
            int y = j + dy[idx];
            if(!isvalid(x, y)) continue;
            swap(cur[i][j], cur[x][y]);
            if (!vis.count(cur)) {
                int h = h2(cur, goal);
                vis.insert(cur);
                q.push({h, {cur, {x, y}}});
            }

            swap(cur[i][j], cur[x][y]);
        }

        moves++;
    }
    return false;
}

// Save the results in file
void saveinFile(vector<int>&vec1, vector<int>&vec2, vector<int>&vec3, vector<int>&vec4) {
    std::ofstream outFile("output.txt");

    if (outFile.is_open()) {
        for (const auto &val : vec1) outFile << val << " ";
        outFile << std::endl;
        for (const auto &val : vec2) outFile << val << " ";
        outFile << std::endl;
        for (const auto &val : vec3) outFile << val << " ";
        outFile << std::endl;
        for (const auto &val : vec4) outFile << val << " ";
        outFile << std::endl;
        outFile.close();
    } else {
        std::cerr << "Unable to open file";
    }
}

// main function
int32_t main(){
    srand(time(0));
    vector<vector<vector<int>>>start, goal;
    vector<vector<int>>tem = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    vector<int>dfsM, bfsM, H1M, H2M;
    vector<double>bfsTime, dfsTime, h1Time, h2Time;
    int temp = 100;

    while(temp--) {
        start.push_back(gen());
        goal.push_back(tem);
    }

    for(int i = 0; i<20; i++) {
        vector<vector<int>> cur = start[i];
        vector<vector<int>> end = goal[i];
        cout<<"Current problem number is: "<<i+1<<endl;
        cout<<"Starting Grid is: "<<endl;
        print(cur);
        cout<<"Ending Grid is: "<<endl;
        print(end);

        if(solvable(cur, end)) {
            bool flag = 0;
            set<vector<vector<int>>> vis;
            for(int i = 0; i<n; i++) {
                for(int k = 0; k<n; k++) {
                    if(cur[i][k] == 0) {
                        flag = 1;

                        // BFS Time Tracking
                        int moves = 0;
                        auto bfsStart = std::chrono::high_resolution_clock::now();
                        if(bfs(cur, end, i, k, moves)) {
                            auto bfsEnd = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> bfsDuration = bfsEnd - bfsStart;
                            bfsTime.push_back(bfsDuration.count());

                            cout<<"Solved Successfully using normal bfs!"<<endl;
                            cout<<"Moves taken to solve: "<<moves<<endl;
                            cout<<"Time taken: "<<bfsDuration.count()<<" seconds"<<endl;
                            bfsM.push_back(moves);
                        } else {
                            cout<<"Failed to solve BFS"<<endl;
                        }

                        // DFS Time Tracking
                        moves = 0;
                        auto dfsStart = std::chrono::high_resolution_clock::now();
                        if(dfs(cur, end, moves)) {
                            auto dfsEnd = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> dfsDuration = dfsEnd - dfsStart;
                            dfsTime.push_back(dfsDuration.count());

                            cout<<"Solved Successfully using normal dfs!"<<endl;
                            cout<<"Moves taken to solve: "<<moves<<endl;
                            cout<<"Time taken: "<<dfsDuration.count()<<" seconds"<<endl;
                            dfsM.push_back(moves);
                        } else {
                            cout<<"Failed to solve DFS"<<endl;
                        }

                        // H1 Best First Search Time Tracking
                        moves = 0;
                        auto h1Start = std::chrono::high_resolution_clock::now();
                        if(bestfsh1(cur, end, i, k, moves)) {
                            auto h1End = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> h1Duration = h1End - h1Start;
                            h1Time.push_back(h1Duration.count());

                            cout<<"Solved Successfully using h1 optimized bfs!"<<endl;
                            cout<<"Moves taken to solve: "<<moves<<endl;
                            cout<<"Time taken: "<<h1Duration.count()<<" seconds"<<endl;
                            H1M.push_back(moves);
                        } else {
                            cout<<"Failed to solve H1"<<endl;
                        }
                        // H2 Best First Search Time Tracking
                        moves = 0;
                        auto h2Start = std::chrono::high_resolution_clock::now();
                        if(bestfsh2(cur, end, i, k, moves)) {
                            auto h2End = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> h2Duration = h2End - h2Start;
                            h2Time.push_back(h2Duration.count());

                            cout<<"Solved Successfully using h2 optimized bfs!"<<endl;
                            cout<<"Moves taken to solve: "<<moves<<endl;
                            cout<<"Time taken: "<<h2Duration.count()<<" seconds"<<endl;
                            H2M.push_back(moves);
                        } else {
                            cout<<"Failed to solve H2"<<endl;
                        }
                    }
                    if(flag) break;
                }
                if(flag) break;
            }
        } else {
            cout<<"NOT SOLVABLE!!"<<endl;
        }
        cout<<endl;
    }
    saveinFile(bfsM, dfsM, H1M, H2M);
    return 0;
}