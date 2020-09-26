#pragma once
template<typename T>
class BTree
{
public:
	T value;
	BTree* Left;
	BTree* Right;
};
