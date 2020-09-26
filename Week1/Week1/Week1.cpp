#include <iostream>
#include "6.h"
#include "5.h"

bool compare(int a, int b) { return a < b; }
int main()
{
	const int max = 50;
	int a[max];
	for (int i = 0; i < max; i++)a[i] = i;
	int* x = static_cast<int*>(TriSearch(a, 0, max, compare));
	std::cout << *x;
}

