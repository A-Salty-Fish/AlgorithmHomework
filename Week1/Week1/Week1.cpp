#include <iostream>
#include "6.h"
#include "5.h"
#include "4.h"

bool compare(int a, int b) { return a < b; }
int main()
{
	const int max = 5;
	int a[max] = { 2,3,4,5,6 };
	
	std::cout << FindSubscript(a, 5);
}

