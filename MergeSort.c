#include <stdio.h>
#include <time.h>

int a[20], n;

void main() {
    int i;
    double time_taken;

    printf("Enter the no. of elements: ");
    scanf("%d", &n);

    printf("Enter the array elements: ");
    for (i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }

    clock_t start, end;
    start = clock();
    merge_sort(a, 0, n - 1);
    end = clock();

    time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Sorted array: ");
    for (i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    printf("Time taken to sort: %f seconds\n", time_taken);
}

void merge_sort(int a[], int low, int high) {
    if (low < high) {
        int mid = low + (high - low) / 2;
        merge_sort(a, low, mid);
        merge_sort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
}

void merge(int a[], int low, int mid, int high) {
    int i = low, j = mid + 1, k = low;
    int c[n];

    while (i <= mid && j <= high) {
        if (a[i] < a[j]) {
            c[k++] = a[i++];
        } 
        else {
            c[k++] = a[j++];
        }
    }

    while (i <= mid) {
        c[k++] = a[i++];
    }

    while (j <= high) {
        c[k++] = a[j++];
    }

    for (i = low; i <= high; i++) {
        a[i] = c[i];
    }
}
