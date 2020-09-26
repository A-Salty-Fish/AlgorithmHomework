#pragma once
#include "BTree.h"
int SumsBiggerNodes(BTree<int>* head, int k)
{
	if (head == nullptr) return 0;
	return SumsBiggerNodes(head->Left, k) +
		SumsBiggerNodes(head->Right, k) +
		head->value >= k ? 1 : 0;
}