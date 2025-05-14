// Write program to obtain the Topological ordering of vertices in a given digraph.
// DFS
#include <stdio.h>

int n, a[10][10], res[10], s[10], top = 0;

int main() {
    printf("Enter the number of nodes: ");
    scanf("%d", &n);

    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &a[i][j]);
        }
    }

    dfs_top(n, a);

    printf("Topological Order: ");
    for (int i = n - 1; i >= 0; i--) {
        printf("%d ", res[i]);
    }
    printf("\n");

    return 0;
}

void dfs_top(int n, int a[][10]) {
    for (int i = 0; i < n; i++) {
        s[i] = 0; // mark all nodes unvisited
    }

    for (int i = 0; i < n; i++) {
        if (s[i] == 0) {
            dfs(i, n, a);
        }
    }
}

void dfs(int j, int n, int a[][10]) {
    s[j] = 1;

    for (int i = 0; i < n; i++) {
        if (a[j][i] == 1 && s[i] == 0) {
            dfs(i, n, a);
        }
    }

    res[top++] = j; // push to result stack
}




// Source Removal
#include <stdio.h>

int a[10][10], n, t[10], indegree[10];
int stack[10], top = -1;


int main() {
    printf("Enter the number of nodes: ");
    scanf("%d", &n);
    
    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &a[i][j]);
        }
    }

    computeIndegree(n, a);
    tps_SourceRemoval(n, a);

    printf("Topological Sort: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", t[i]);
    }
    printf("\n");

    return 0;
}

void computeIndegree(int n, int a[][10]) {
    int i, j, sum;
    for (i = 0; i < n; i++) {
        sum = 0;
        for (j = 0; j < n; j++) {
            sum += a[j][i]; // Sum the incoming edges (indegree)
        }
        indegree[i] = sum;
    }
}

void tps_SourceRemoval(int n, int a[][10]) {
    int i, j, v;
    
    // Add nodes with indegree 0 to the stack
    for (i = 0; i < n; i++) {
        if (indegree[i] == 0) {
            stack[++top] = i;
        }
    }

    int k = 0;
    while (top != -1) {
        v = stack[top--]; // Pop from the stack
        t[k++] = v; // Add to the topological order
        for (i = 0; i < n; i++) {
            if (a[v][i] != 0) { // If there is an edge v -> i
                indegree[i]--;
                if (indegree[i] == 0) {
                    stack[++top] = i; // If indegree becomes 0, add it to stack
                }
            }
        }
    }
}




// Implement Johnson Trotter algorithm to generate permutations.
#include <stdio.h>
#include <stdlib.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void generatePermutations(int arr[], int start, int end) {
    if (start == end) {
        for (int i = 0; i <= end; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    } else {
        for (int i = start; i <= end; i++) {
            swap(&arr[start], &arr[i]);
            generatePermutations(arr, start + 1, end);
            swap(&arr[start], &arr[i]); // backtrack
        }
    }
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int* arr = (int*)malloc(n * sizeof(int));
    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    printf("All permutations:\n");
    generatePermutations(arr, 0, n - 1);

    free(arr);
    return 0;
}



// Merge Sort
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int N;

    printf("Enter the number of elements: ");
    scanf("%d", &N);

    int arr[N];
    
    printf("Enter the elements:\n");
    for (int i = 0; i < N; i++) {
        scanf("%d", &arr[i]);
    }

    clock_t start = clock();
    mergeSort(arr, 0, N - 1);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Sorted array: \n");
    printArray(arr, N);

    printf("Time taken for merge sort: %f seconds\n", time_taken);

    return 0;
}




// Quick Sort
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1;  // Return the partition index
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high); 
        quickSort(arr, low, pi - 1);  
        quickSort(arr, pi + 1, high);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int N;

    printf("Enter the number of elements: ");
    scanf("%d", &N);

    int arr[N];

    printf("Enter the elements:\n");
    for (int i = 0; i < N; i++) {
        scanf("%d", &arr[i]);
    }

    clock_t start = clock();
    quickSort(arr, 0, N - 1);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Sorted array: \n");
    printArray(arr, N);

    printf("Time taken for quick sort: %f seconds\n", time_taken);

    return 0;
}



// Heap Sort
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void heapify(int arr[], int n, int i) {
    int largest = i;    
    int left = 2 * i + 1; 
    int right = 2 * i + 2; 

    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }

    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }

    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;

        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    for (int i = n - 1; i > 0; i--) {
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        heapify(arr, i, 0);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int N;

    printf("Enter the number of elements: ");
    scanf("%d", &N);

    int arr[N];

    printf("Enter the elements:\n");
    for (int i = 0; i < N; i++) {
        scanf("%d", &arr[i]);
    }

    clock_t start = clock();
    heapSort(arr, N);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Sorted array: \n");
    printArray(arr, N);

    printf("Time taken for heap sort: %f seconds\n", time_taken);

    return 0;
}



// 0/1 Knapsack - DP
#include <stdio.h>
#include <stdlib.h>

int knapsack(int W, int n, int weights[], int values[]) {
    int dp[n+1][W+1];

    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (i == 0 || w == 0) {
                dp[i][w] = 0;
            } 
            else if (weights[i-1] <= w) {
                dp[i][w] = (values[i-1] + dp[i-1][w - weights[i-1]] > dp[i-1][w]) ?
                            values[i-1] + dp[i-1][w - weights[i-1]] : dp[i-1][w];
            } 
            else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }

    return dp[n][W];
}

int main() {
    int n, W;

    printf("Enter the number of items: ");
    scanf("%d", &n);
    printf("Enter the knapsack capacity: ");
    scanf("%d", &W);

    int weights[n], values[n];

    printf("Enter the weights of the items: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &weights[i]);
    }
    printf("Enter the values of the items: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &values[i]);
    }

    int maxValue = knapsack(W, n, weights, values);

    printf("Maximum value that can be obtained: %d\n", maxValue);

    return 0;
}



// Floyd Algo
#include <stdio.h>
#include <stdlib.h>

#define INF 99999

void floydWarshall(int graph[][4], int V) {
    int dist[V][V];

    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            dist[i][j] = graph[i][j];
        }
    }

    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    printf("The shortest distances between every pair of vertices:\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == INF) {
                printf("%4s ", "INF");
            } else {
                printf("%4d ", dist[i][j]);
            }
        }
        printf("\n");
    }
}

int main() {
    int V = 4; 

    int graph[4][4] = {
        {0, 3, INF, INF},
        {2, 0, INF, INF},
        {INF, 7, 0, 1},
        {6, INF, INF, 0}
    };

    floydWarshall(graph, V);

    return 0;
}



// Prim's Algorithm
#include <stdio.h>
#include <limits.h>

#define MAX_VERTICES 10
#define INF INT_MAX

int minKey(int key[], int mstSet[], int V) {
    int min = INF, min_index;
    
    for (int v = 0; v < V; v++) {
        if (mstSet[v] == 0 && key[v] < min) {
            min = key[v];
            min_index = v;
        }
    }
    return min_index;
}

void primMST(int graph[MAX_VERTICES][MAX_VERTICES], int V) {
    int parent[V];
    int key[V];    
    int mstSet[V];

    for (int i = 0; i < V; i++) {
        key[i] = INF;
        mstSet[i] = 0;
    }

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet, V);

        mstSet[u] = 1;

        for (int v = 0; v < V; v++) {
            if (graph[u][v] != 0 && mstSet[v] == 0 && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    printf("Edge \tWeight\n");
    int totalCost = 0;
    for (int i = 1; i < V; i++) {
        printf("%d - %d \t%d\n", parent[i], i, graph[i][parent[i]]);
        totalCost += graph[i][parent[i]];
    }

    printf("Total cost of MST: %d\n", totalCost);
}

int main() {
    int V, graph[MAX_VERTICES][MAX_VERTICES];

    printf("Enter the number of vertices: ");
    scanf("%d", &V);

    printf("Enter the adjacency matrix (0 for no edge):\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            scanf("%d", &graph[i][j]);
        }
    }

    primMST(graph, V);

    return 0;
}




// Kruskal Algorithm
#include <stdio.h>
#include <stdlib.h>

#define MAX_EDGES 50
#define MAX_VERTICES 10
#define INF 999999

typedef struct {
    int u, v, weight;
} Edge;

int parent[MAX_VERTICES], rank[MAX_VERTICES];

int find(int x) {
    if (parent[x] == x)
        return x;
    return parent[x] = find(parent[x]);
}

void unionSets(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);

    if (rootX != rootY) {
        if (rank[rootX] < rank[rootY])
            parent[rootX] = rootY;
        else if (rank[rootX] > rank[rootY])
            parent[rootY] = rootX;
        else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
}

int compare(const void* a, const void* b) {
    return ((Edge*)a)->weight - ((Edge*)b)->weight;
}

void kruskalMST(Edge edges[], int E, int V) {
    for (int i = 0; i < V; i++) {
        parent[i] = i;
        rank[i] = 0;
    }

    qsort(edges, E, sizeof(Edge), compare);

    int mstWeight = 0;
    printf("Edges in the MST:\n");

    for (int i = 0; i < E; i++) {
        int u = edges[i].u;
        int v = edges[i].v;

        if (find(u) != find(v)) {
            printf("%d - %d : %d\n", u, v, edges[i].weight);
            mstWeight += edges[i].weight;
            unionSets(u, v);
        }
    }

    printf("Total cost of MST: %d\n", mstWeight);
}

int main() {
    int V, E;
    Edge edges[MAX_EDGES];

    printf("Enter the number of vertices: ");
    scanf("%d", &V);
    printf("Enter the number of edges: ");
    scanf("%d", &E);

    printf("Enter the edges (u, v, weight):\n");
    for (int i = 0; i < E; i++) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].weight);
    }

    kruskalMST(edges, E, V);

    return 0;
}




// Fractional Knapsack - Greedy
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int weight;
    int value;
    float ratio; 
} Item;

int compare(const void *a, const void *b) {
    float ratioA = ((Item *)a)->ratio;
    float ratioB = ((Item *)b)->ratio;
    if (ratioA > ratioB)
        return -1;
    else if (ratioA < ratioB)
        return 1;
    else
        return 0;
}

float fractionalKnapsack(Item items[], int n, int capacity) {
    qsort(items, n, sizeof(Item), compare);

    int currentWeight = 0;
    float totalValue = 0.0;

    for (int i = 0; i < n; i++) {
        if (currentWeight + items[i].weight <= capacity) {
            currentWeight += items[i].weight;
            totalValue += items[i].value;
        } 
        else {
            int remainingWeight = capacity - currentWeight;
            totalValue += items[i].value * ((float)remainingWeight / items[i].weight);
            break;
        }
    }

    return totalValue;
}

int main() {
    int n, capacity;

    printf("Enter the number of items: ");
    scanf("%d", &n);

    printf("Enter the capacity of the knapsack: ");
    scanf("%d", &capacity);

    Item items[n];

    printf("Enter the weight and value for each item:\n");
    for (int i = 0; i < n; i++) {
        printf("Item %d - Weight: ", i + 1);
        scanf("%d", &items[i].weight);
        printf("Item %d - Value: ", i + 1);
        scanf("%d", &items[i].value);
        items[i].ratio = (float)items[i].value / items[i].weight;
    }

    float maxValue = fractionalKnapsack(items, n, capacity);

    printf("The maximum value that can be obtained is: %.2f\n", maxValue);

    return 0;
}



// Dijkstra
#include <stdio.h>
#include <limits.h>

#define MAX_VERTICES 10
#define INF INT_MAX

int minDistance(int dist[], int visited[], int V) {
    int min = INF, min_index;
    for (int v = 0; v < V; v++) {
        if (visited[v] == 0 && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

void dijkstra(int graph[MAX_VERTICES][MAX_VERTICES], int V, int source) {
    int dist[MAX_VERTICES]; 
    int visited[MAX_VERTICES] = {0}; 

    for (int i = 0; i < V; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, visited, V);
        
        visited[u] = 1;

        for (int v = 0; v < V; v++) {
            if (!visited[v] && graph[u][v] != 0 && dist[u] != INF && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    printf("Vertex\t\tDistance from Source\n");
    for (int i = 0; i < V; i++) {
        printf("%d\t\t%d\n", i, dist[i]);
    }
}

int main() {
    int V, source;

    printf("Enter the number of vertices: ");
    scanf("%d", &V);

    int graph[MAX_VERTICES][MAX_VERTICES];

    printf("Enter the adjacency matrix (use 0 for no edge between vertices):\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            scanf("%d", &graph[i][j]);
        }
    }

    printf("Enter the source vertex (0 to %d): ", V - 1);
    scanf("%d", &source);

    dijkstra(graph, V, source);

    return 0;
}





// N queens
#include <stdio.h>
#include <conio.h>
#include <math.h>
int x[20], count = 1;
void queens(int, int);
int place(int, int);

void main()
{
    int n, k = 1;
    printf("\n enter the number of queens to be placed\n");
    scanf("%d", &n);
    queens(k, n);
}

void queens(int k, int n)
{
    int i, j;
    for (j = 1; j <= n; j++)
    {
        if (place(k, j))
        {
            x[k] = j;
            if (k == n)
            {
                printf("\n %d solution", count);
                count++;
                for (i = 1; i <= n; i++)
                    printf("\n \t %d row <---> %d column", i, x[i]);
            }
            else
                queens(k + 1, n);
        }
    }
}
int place(int k, int j)
{
    int i;
    for (i = 1; i < k; i++)
    {
        if ((x[i] == j) || (abs(x[i] - j)) == abs(i - k))
            return 0;
    }
    return 1;
}
