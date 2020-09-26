#pragma once
template<typename T>
T* TriSearch(T* First, T target, int length, bool (*compare)(T a,T b))
{
	T* Second = First + length / 3;
	T* Third = First + length * 2 / 3;
	T* End = First + length;
	if (length <= 2)
		return *First == target ? First : *(End - 1) == target ? End - 1 : nullptr;
	if (compare(target, *First) || compare(*(End - 1), target))
		return nullptr;
	if (compare(target, *Second))
		return TriSearch(First, target, length / 3, compare);
	else if (compare(target,*Third))
		return TriSearch(Second, target, Third - Second, compare);
	else
		return TriSearch(Third, target, End - Third, compare);
}