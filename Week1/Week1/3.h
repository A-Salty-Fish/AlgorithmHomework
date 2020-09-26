#pragma once
#include "BTree.h"
template<typename T>
int GetDepth(BTree<T>* head, T x)
{
	if (head == nullptr) return 0;
	if (head->value == x) return 1;
	int leftDepth = GetDepth(head->Left, x);
	if (leftDepth > 0) return leftDepth + 1;
	int rightDepth = GetDepth(head->Right, x);
	if (rightDepth > 0) return rightDepth + 1;
	return 0;
}