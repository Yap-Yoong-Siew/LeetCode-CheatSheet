**Python Competitive Programming & Interview Cheatsheet**

---

## **Input Parsing**
```python
# Basic inputs
n = int(input())                        # Single integer
a, b, c = map(int, input().split())     # Multiple integers on one line
arr = list(map(int, input().split()))   # List of integers on one line
s = input()                             # String input
x, y = map(float, input().split())      # Floating point numbers

# Multiple lines
n = int(input())                        # Number of lines
lines = [input() for _ in range(n)]     # Read n lines as strings
grid = [list(input()) for _ in range(n)]  # Read n lines as character grid
matrix = [list(map(int, input().split())) for _ in range(n)]  # n x m matrix of integers

# Fixed-format input
for _ in range(int(input())):           # t test cases
    n, k = map(int, input().split())    # Parse first line of each test
    arr = list(map(int, input().split()))  # Parse second line
    # Process test case

# Continuous input with known number of lines
n, q = map(int, input().split())        # n elements, q queries
arr = list(map(int, input().split()))
for _ in range(q):
    query = list(map(int, input().split()))
    # Process each query

# Reading until EOF
import sys
lines = sys.stdin.readlines()           # Read all lines at once
# OR line by line
while True:
    try:
        line = input()
        # Process line
    except EOFError:
        break

# Fast I/O for large inputs
import sys
input = sys.stdin.readline              # Much faster for large inputs
n = int(input())                        # Remember to strip() if needed
a, b = map(int, input().strip().split())
```

## **Output Formatting**
```python
# Basic output
print(answer)                           # Simple output
print(f"{answer:.6f}")                  # Print with 6 decimal places
print(*arr)                             # Print space-separated array
print('\n'.join(map(str, arr)))         # Print array with each element on new line

# Formatted output
print(f"Case #{t+1}: {answer}")         # Common in competitions
print("YES" if condition else "NO")     # Conditional output
```

---

## **Data Structure Complexity**

| Data Structure | Access | Search | Insert | Delete | Space |
|----------------|--------|--------|--------|--------|-------|
| List           | O(1)   | O(n)   | O(1)*  | O(n)   | O(n)  |
| Dictionary     | O(1)   | O(1)   | O(1)   | O(1)   | O(n)  |
| Set            | N/A    | O(1)   | O(1)   | O(1)   | O(n)  |
| Tuple          | O(1)   | O(n)   | N/A    | N/A    | O(n)  |
| Heap           | N/A    | O(n)   | O(log n) | O(log n) | O(n) |
| Deque          | O(1)   | O(n)   | O(1)   | O(1)   | O(n)  |

*Amortized time complexity

## **Data Structures**

### **Lists**
```python
lst = [1, 2, 3]
lst.append(4)      # [1,2,3,4]
lst.insert(1, 5)   # [1,5,2,3,4]
lst.pop()          # [1,5,2,3]
lst.remove(5)      # [1,2,3]
lst.index(2)       # 1 (index of first occurrence)
lst.count(1)       # 1 (number of occurrences)
lst.sort()         # Sort in-place
lst.reverse()      # Reverse in-place
lst = lst[::-1]    # Reverse (creates new list)
```

### **Dictionaries**
```python
d = {'a': 1, 'b': 2}
d['c'] = 3
d.get('z', 0)      # 0 (default)
d.items()          # dict_items([('a',1),('b',2),('c',3)])
d.keys()           # dict_keys(['a', 'b', 'c'])
d.values()         # dict_values([1, 2, 3])
d.pop('b')         # Removes key 'b' and returns its value
d.setdefault('d', 4)  # Sets d['d']=4 if 'd' not in d
```

### **Sets**
```python
s = {1, 2, 3}
s.add(4)           # {1, 2, 3, 4}
s.remove(2)        # {1, 3, 4} - raises KeyError if not found
s.discard(2)       # Removes if present, no error if not found
s1, s2 = {1, 2}, {2, 3}
s1 & s2            # {2} (intersection)
s1 | s2            # {1,2,3} (union)
s1 - s2            # {1} (difference)
s1 ^ s2            # {1,3} (symmetric difference)
```

### **Priority Queue (Min Heap)**
```python
import heapq
pq = []
heapq.heappush(pq, 5)
smallest = heapq.heappop(pq)
# For max heap, negate values
heapq.heappush(pq, -5)  # Push 5 as -5
largest = -heapq.heappop(pq)  # Pop -5 and negate to get 5

# Convert list to heap in-place
nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)  # nums is now [1, 1, 4, 3, 5]

# Get n smallest/largest elements
smallest_three = heapq.nsmallest(3, nums)
largest_three = heapq.nlargest(3, nums)
```

### **Deque (Efficient Queue)**
```python
from collections import deque
q = deque()
q.append(1)      # add to right
q.appendleft(0)  # add to left
q.popleft()      # remove from left
q.pop()          # remove from right
q.extend([2,3])  # add multiple to right
q.extendleft([2,3])  # add multiple to left (3 will be before 2)
q.rotate(1)      # rotate right by 1
q.rotate(-1)     # rotate left by 1
```

### **Counter**
```python
from collections import Counter
cnt = Counter([1,1,2,3])  # {1:2, 2:1, 3:1}
cnt.most_common(2)        # [(1, 2), (2, 1)]
cnt[1] += 1               # Increment count
cnt.update([1,2,2])       # Add multiple occurrences
cnt.elements()            # Iterator over elements, repeating each as many times as its count
list(cnt.elements())      # [1, 1, 1, 2, 2, 3]
```

### **DefaultDict**
```python
from collections import defaultdict
graph = defaultdict(list)  # default value is empty list
graph[1].append(2)         # No KeyError if key doesn't exist
visit_count = defaultdict(int)  # default value is 0
visit_count['node'] += 1   # Increment counter

# Other useful defaults
dd_set = defaultdict(set)
dd_dict = defaultdict(dict)
dd_lambda = defaultdict(lambda: [0, 0])  # Custom default
```

### **OrderedDict**
```python
from collections import OrderedDict
od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3
od.move_to_end('a')  # Move to end
od.popitem(last=False)  # Remove from beginning
```

---

## **String Operations**
```python
s = "hello world"
s.upper()                # "HELLO WORLD"
s.lower()                # "hello world"
s.startswith("he")       # True
s.endswith("ld")         # True
s.replace("world", "python")  # "hello python"
s.split()                # ["hello", "world"]
s.strip()                # Remove whitespace from both ends
" ".join(["a", "b"])     # "a b"
"".join(reversed(s))     # Reverse a string
s.find("world")          # 6 (index of first occurrence, -1 if not found)
s.count("l")             # 3 (number of occurrences)
s.isalpha()              # False (checks if all chars are alphabetic)
s.isdigit()              # False (checks if all chars are digits)
s.isalnum()              # False (checks if all chars are alphanumeric)
```

---

## **Algorithms**

### **Searching**
```python
# Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1  # Not found

# Binary Search using bisect
import bisect
def binary_search_bisect(arr, target):
    idx = bisect.bisect_left(arr, target)
    return idx if idx < len(arr) and arr[idx] == target else -1

# Find insertion point (index where element should be inserted to maintain order)
idx = bisect.bisect_left(arr, target)  # First position where target could be inserted
idx = bisect.bisect_right(arr, target)  # Last position where target could be inserted
```

### **Sorting**
```Python
# Sorting lists
sorted([3,1,2])         # [1,2,3]
sorted([3,1,2], reverse=True)  # [3,2,1]
sorted([(1,'b'), (2,'a')], key=lambda x: x[1])  # [(2,'a'), (1,'b')]

# Sort in-place
arr = [3,1,2]
arr.sort()  # arr is now [1,2,3]

# Sort by multiple criteria
students = [('Alice', 90), ('Bob', 85), ('Charlie', 90)]
sorted(students, key=lambda x: (-x[1], x[0]))  # Sort by grade desc, then name asc
```

```python
# Quick Sort
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

### **Graph Algorithms**
```python
# BFS
from collections import deque
def bfs(graph, start):
    visited, queue = set([start]), deque([start])
    while queue:
        vertex = queue.popleft()
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

# BFS with path tracking
def bfs_path(graph, start, end):
    queue = deque([(start, [start])])  # (node, path)
    visited = set([start])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None  # No path found
```

```python
# DFS
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# DFS with path tracking
def dfs_path(graph, start, end, path=None, visited=None):
    if path is None:
        path = [start]
    if visited is None:
        visited = set([start])
    if start == end:
        return path
    for neighbor in graph[start]:
        if neighbor not in visited:
            visited.add(neighbor)
            result = dfs_path(graph, neighbor, end, path + [neighbor], visited)
            if result:
                return result
    return None  # No path found
```

```python
# Dijkstra's Algorithm (Shortest Path)
def dijkstra(graph, start):
    import heapq
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current_dist > distances[current]:
            continue
        for neighbor, weight in graph[current].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances
```

```python
# Union-Find (Disjoint Set)
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of connected components
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
        
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
        
        self.count -= 1  # Decrease connected components
        return True
```

### **Dynamic Programming**
```python
# Fibonacci with Memoization
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

# Bottom-up DP (Tabulation)
def fibonacci_bottom_up(n):
    if n <= 2:
        return 1
    dp = [0] * (n + 1)
    dp[1] = dp[2] = 1
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

### **Kadane's Algorithm (Max Subarray Sum)**
```python
def max_subarray_sum(arr):
    max_so_far = max_ending_here = arr[0]
    for x in arr[1:]:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

# With subarray indices
def max_subarray_with_indices(arr):
    max_so_far = arr[0]
    max_ending_here = arr[0]
    start = end = s = 0
    
    for i in range(1, len(arr)):
        if max_ending_here + arr[i] > arr[i]:
            max_ending_here += arr[i]
        else:
            max_ending_here = arr[i]
            s = i
            
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = s
            end = i
            
    return max_so_far, start, end
```

### **Binary Search on Answer**
```python
def possible(mid, arr, target):
    # Check if 'mid' works as a solution
    # Implementation depends on the problem
    return True/False

def binary_search_answer(arr, target):
    left, right = min_val, max_val
    while left <= right:
        mid = (left + right) // 2
        if possible(mid, arr, target):
            right = mid - 1  # For finding minimum valid value
            # OR left = mid + 1  # For finding maximum valid value
        else:
            left = mid + 1   # For finding minimum valid value
            # OR right = mid - 1  # For finding maximum valid value
    return left  # Or another appropriate value
```

---

## **Math & Number Theory**
```python
# GCD and LCM
from math import gcd
def lcm(a, b):
    return a * b // gcd(a, b)

# Extended Euclidean Algorithm (ax + by = gcd(a,b))
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

# Check if prime
def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Sieve of Eratosthenes (find all primes up to n)
def sieve_of_eratosthenes(n):
    primes = [True for i in range(n+1)]
    p = 2
    while p * p <= n:
        if primes[p]:
            for i in range(p * p, n+1, p):
                primes[i] = False
        p += 1
    return [p for p in range(2, n+1) if primes[p]]

# Power with modulo
def pow_mod(x, y, m):
    if y == 0: return 1
    p = pow_mod(x, y // 2, m) % m
    p = (p * p) % m
    return p if y % 2 == 0 else (x * p) % m

# Factorial with modulo
def factorial_mod(n, mod):
    result = 1
    for i in range(2, n + 1):
        result = (result * i) % mod
    return result

# Combinations with modulo
def comb_mod(n, k, mod):
    # Calculate C(n,k) % mod
    num = factorial_mod(n, mod)
    denom = (factorial_mod(k, mod) * factorial_mod(n - k, mod)) % mod
    return (num * pow_mod(denom, mod - 2, mod)) % mod  # Using Fermat's little theorem
```

---

## **Bit Manipulation**
```python
# Set/unset/check bit
def set_bit(x, position): return x | (1 << position)
def clear_bit(x, position): return x & ~(1 << position)
def check_bit(x, position): return (x >> position) & 1
def toggle_bit(x, position): return x ^ (1 << position)

# Count set bits
bin(n).count('1')  # Python way
def count_set_bits(n):
    count = 0
    while n:
        n &= n - 1  # Clear the least significant bit set
        count += 1
    return count

# Common bit operations
x & y   # AND: bits set in both x and y
x | y   # OR: bits set in either x or y
x ^ y   # XOR: bits set in either x or y but not both
~x      # NOT: flip all bits
x << y  # Left shift: multiply x by 2^y
x >> y  # Right shift: divide x by 2^y

# Useful bit tricks
n & (n-1)  # Clear the lowest set bit
n & -n     # Keep only the lowest set bit
n | (n-1)  # Set all the bits below the lowest set bit
```

---

## **Common Patterns**

### **Two Pointers**
```python
def two_sum(nums, target):
    nums.sort()
    left, right = 0, len(nums) - 1
    while left < right:
        curr_sum = nums[left] + nums[right]
        if curr_sum == target:
            return True
        if curr_sum < target:
            left += 1
        else:
            right -= 1
    return False

# Two pointers for removing duplicates
def remove_duplicates(nums):
    if not nums:
        return 0
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1
```

### **Sliding Window**
```python
def max_subarray_sum(arr, k):
    max_sum = current_sum = sum(arr[:k])
    for i in range(k, len(arr)):
        current_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, current_sum)
    return max_sum

# Variable size sliding window
def smallest_subarray_with_sum(arr, target_sum):
    window_sum = 0
    min_length = float('inf')
    window_start = 0
    
    for window_end in range(len(arr)):
        window_sum += arr[window_end]
        
        while window_sum >= target_sum:
            min_length = min(min_length, window_end - window_start + 1)
            window_sum -= arr[window_start]
            window_start += 1
            
    return min_length if min_length != float('inf') else 0
```

### **Prefix Sum**
```python
def prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

# Query sum of range [i, j] inclusive
def range_sum(prefix, i, j):
    return prefix[j + 1] - prefix[i]

# 2D Prefix Sum
def prefix_sum_2d(matrix):
    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    for i in range(rows):
        for j in range(cols):
            prefix[i + 1][j + 1] = prefix[i + 1][j] + prefix[i][j + 1] - prefix[i][j] + matrix[i][j]
    
    return prefix

# Query sum of submatrix [(r1,c1), (r2,c2)] inclusive
def submatrix_sum(prefix, r1, c1, r2, c2):
    return prefix[r2 + 1][c2 + 1] - prefix[r2 + 1][c1] - prefix[r1][c2 + 1] + prefix[r1][c1]
```

### **Greedy Algorithms**
```python
# Activity Selection
def activity_selection(start, finish):
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    selected = [activities[0]]
    
    for activity in activities[1:]:
        if activity[0] >= selected[-1][1]:
            selected.append(activity)
            
    return selected

# Fractional Knapsack
def fractional_knapsack(values, weights, capacity):
    items = sorted(zip(values, weights), key=lambda x: x[0]/x[1], reverse=True)
    total_value = 0
    
    for value, weight in items:
        if capacity >= weight:
            total_value += value
            capacity -= weight
        else:
            total_value += value * (capacity / weight)
            break
            
    return total_value
```

---

## **Python Tricks**

### **List Comprehensions**
```python
[x**2 for x in range(10) if x % 2 == 0]  # [0,4,16,36,64]

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]  # [1,2,3,4,5,6,7,8,9]

# Dictionary comprehension
{x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}

# Set comprehension
{x % 3 for x in range(10)}  # {0, 1, 2}
```

### **Lambda Functions**
```python
sorted([(1,'b'), (5,'a'), (3,'c')], key=lambda x: x[1])  # [(5,'a'),(1,'b'),(3,'c')]

# Multiple criteria sorting
sorted([(1,'b'), (5,'a'), (3,'a')], key=lambda x: (x[1], x[0]))  # [(3,'a'),(5,'a'),(1,'b')]

# Map with lambda
list(map(lambda x: x**2, range(5)))  # [0, 1, 4, 9, 16]

# Filter with lambda
list(filter(lambda x: x % 2 == 0, range(10)))  # [0, 2, 4, 6, 8]
```

### **Useful Built-ins**
```python
from collections import Counter, defaultdict, deque, OrderedDict
from itertools import permutations, combinations, product, accumulate, groupby
from functools import lru_cache, reduce
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapify, nlargest, nsmallest

# zip to transpose a matrix
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = list(zip(*matrix))  # [(1, 4), (2, 5), (3, 6)]

# enumerate for index and value
for i, val in enumerate(['a', 'b', 'c']):
    print(i, val)  # 0 a, 1 b, 2 c

# all and any
all([True, True, False])  # False
any([True, False, False])  # True

# sum, min, max with key function
words = ['apple', 'banana', 'cherry']
max(words, key=len)  # 'banana'
min(words, key=lambda x: x[1])  # 'cherry' (min by second letter)

# sorted with custom key
sorted("This is a test string".split(), key=str.lower)
```

### **Itertools (Combinations & Permutations)**
```python
from itertools import permutations, combinations, combinations_with_replacement, product

list(permutations([1,2,3], 2))  # [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)]
list(combinations([1,2,3], 2))  # [(1,2), (1,3), (2,3)]
list(combinations_with_replacement([1,2], 2))  # [(1,1), (1,2), (2,2)]
list(product([1,2], [3,4]))  # [(1,3), (1,4), (2,3), (2,4)]

# Other useful itertools
from itertools import count, cycle, repeat, chain, islice, groupby, accumulate

# count: infinite counter
for i in islice(count(10, 2), 5):  # Start at 10, step 2, take 5 elements
    print(i)  # 10, 12, 14, 16, 18

# accumulate: running sum
list(accumulate([1, 2, 3, 4, 5]))  # [1, 3, 6, 10, 15]

# groupby: group consecutive elements
[k for k, g in groupby('AAAABBBCCDAABBB')]  # ['A', 'B', 'C', 'D', 'A', 'B']
```

### **Memoization with functools**
```python
from functools import lru_cache

@lru_cache(maxsize=None)  # Unlimited cache size
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Clear cache if needed
fibonacci.cache_clear()
```

---

## **Advanced Data Structures**

### **Trie (Prefix Tree)**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### **Segment Tree**
```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        # Size of segment tree is 2*2^(log2(n)) - 1
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        if start == end:
            # Leaf node
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            # Recursively build left and right subtrees
            self.build(arr, 2 * node + 1, start, mid)
            self.build(arr, 2 * node + 2, mid + 1, end)
            # Internal node stores sum of its children
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def update(self, index, value, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1
        
        if start == end:
            # Leaf node
            self.tree[node] = value
            return
        
        mid = (start + end) // 2
        if index <= mid:
            # Update in left subtree
            self.update(index, value, 2 * node + 1, start, mid)
        else:
            # Update in right subtree
            self.update(index, value, 2 * node + 2, mid + 1, end)
        
        # Update current node
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def query(self, left, right, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1
        
        # No overlap
        if start > right or end < left:
            return 0
        
        # Complete overlap
        if start >= left and end <= right:
            return self.tree[node]
        
        # Partial overlap - query both children
        mid = (start + end) // 2
        left_sum = self.query(left, right, 2 * node + 1, start, mid)
        right_sum = self.query(left, right, 2 * node + 2, mid + 1, end)
        
        return left_sum + right_sum
```

### **Fenwick Tree (Binary Indexed Tree)**
```python
class FenwickTree:
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)
    
    def update(self, idx, delta):
        """Add delta to element at index idx (1-based indexing)"""
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx  # Add least significant bit
    
    def query(self, idx):
        """Get sum of elements from 1 to idx (inclusive)"""
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & -idx  # Remove least significant bit
        return result
    
    def range_query(self, left, right):
        """Get sum of elements from left to right (inclusive)"""
        return self.query(right) - self.query(left - 1)

# Example usage
def create_fenwick(arr):
    n = len(arr)
    fenwick = FenwickTree(n)
    for i in range(n):
        fenwick.update(i + 1, arr[i])  # 1-based indexing
    return fenwick
```

---

## **Debugging Techniques**

### **Print Debugging**
```python
# Add debug prints
def debug_print(*args, **kwargs):
    print("DEBUG:", *args, **kwargs)

# For competitive programming, use this to quickly enable/disable debug prints
DEBUG = True
def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Print variables with their names
a = 5
b = [1, 2, 3]
debug(f"a = {a}, b = {b}")

# Print array with indices
def print_array_with_indices(arr):
    for i, val in enumerate(arr):
        debug(f"arr[{i}] = {val}")
```

### **Visualizing Data Structures**
```python
# Visualize a graph
def print_graph(graph):
    for node, neighbors in graph.items():
        print(f"Node {node} -> {neighbors}")

# Visualize a matrix
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))

# Visualize a tree (simple version)
def print_tree(root, level=0):
    if root is None:
        return
    print("  " * level + str(root.val))
    for child in root.children:
        print_tree(child, level + 1)
```

### **Testing and Edge Cases**
```python
# Test with small examples
test_cases = [
    ([1, 2, 3], 6),
    ([1, 2, 3, 4], 10),
    ([], 0),
    ([5], 5)
]

for arr, expected in test_cases:
    result = sum(arr)
    assert result == expected, f"Failed for {arr}: got {result}, expected {expected}"

# Common edge cases to consider:
# - Empty input
# - Single element input
# - Very large input
# - Input with duplicates
# - Input with negative numbers
# - Minimum and maximum possible values
# - Sorted vs unsorted input
```

---

## **Common Problem Types**

### **1. Array/String Problems**
```python
# Two Sum
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Longest Substring Without Repeating Characters
def length_of_longest_substring(s):
    char_dict = {}
    max_length = start = 0
    
    for i, char in enumerate(s):
        if char in char_dict and start <= char_dict[char]:
            start = char_dict[char] + 1
        else:
            max_length = max(max_length, i - start + 1)
        char_dict[char] = i
    
    return max_length
```

### **2. Dynamic Programming Problems**
```python
# Knapsack Problem
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# Longest Common Subsequence
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

### **3. Graph Problems**
```python
# Detect Cycle in Undirected Graph
def has_cycle_undirected(graph):
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    return False

# Topological Sort
def topological_sort(graph):
    visited = set()
    temp = set()
    order = []
    
    def dfs(node):
        if node in temp:
            return False  # Cycle detected
        if node in visited:
            return True
        
        temp.add(node)
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        
        temp.remove(node)
        visited.add(node)
        order.append(node)
        return True
    
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []  # Cycle detected, no valid topological sort
    
    return order[::-1]
```

### **4. Tree Problems**
```python
# Tree Traversals
def inorder(root):
    result = []
    if root:
        result.extend(inorder(root.left))
        result.append(root.val)
        result.extend(inorder(root.right))
    return result

def preorder(root):
    result = []
    if root:
        result.append(root.val)
        result.extend(preorder(root.left))
        result.extend(preorder(root.right))
    return result

def postorder(root):
    result = []
    if root:
        result.extend(postorder(root.left))
        result.extend(postorder(root.right))
        result.append(root.val)
    return result

# Level Order Traversal
def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

---

## **Handling Large Inputs/Outputs**

### **Fast I/O**
```python
import sys
input = sys.stdin.readline
print = sys.stdout.write

# For reading integers
def read_int():
    return int(input())

def read_ints():
    return list(map(int, input().split()))

# For writing output efficiently
def write(s):
    print(str(s) + '\n')
    
# For competitive programming, sometimes it's useful to buffer output
def buffered_output():
    output = []
    # ... add strings to output list ...
    sys.stdout.write('\n'.join(output))
```

### **Memory Optimization**
```python
# Use generators for large sequences
def large_range(n):
    i = 0
    while i < n:
        yield i
        i += 1

# Process large files line by line
def process_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            # Process line
            pass
```

---

## **Common Python Pitfalls**

### **Mutable Default Arguments**
```python
# Wrong:
def append_to(element, to=[]):
    to.append(element)
    return to

# Right:
def append_to(element, to=None):
    if to is None:
        to = []
    to.append(element)
    return to
```

### **Late Binding Closures**
```python
# Wrong:
funcs = [lambda x: i * x for i in range(5)]

# Right:
funcs = [lambda x, i=i: i * x for i in range(5)]
```

### **Modifying Lists During Iteration**
```python
# Wrong:
numbers = [1, 2, 3, 4, 5]
for i in range(len(numbers)):
    if numbers[i] % 2 == 0:
        numbers.pop(i)  # This will cause issues!

# Right:
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
```

---

This cheatsheet covers essential concepts for competitive programming and technical interviews!
