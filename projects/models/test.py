def find_identical_values_and_remove(lst):
    seen = set()
    identical_indices = []

    for i, value in reversed(list(enumerate(lst))):
        if value in seen:
            identical_indices.append(i)
            del lst[i]
        else:
            seen.add(value)

    return identical_indices


my_list = [10, 2, 3, 4, 2]
print("List before removal:", my_list)
identical_indices = find_identical_values_and_remove(my_list)

print("Indices of identical values:", identical_indices)
print("List after removal:", my_list)