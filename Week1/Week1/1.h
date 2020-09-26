#pragma once
#include "BTree.h"
int SumsLeafNodes(BTree<int>* head)
{
	if (head == nullptr) return 0;
	if (head->Left == nullptr && head->Right == nullptr) 
		return head->value;
	return SumsLeafNodes(head->Left) + SumsLeafNodes(head->Right);
}