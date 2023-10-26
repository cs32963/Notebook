# Craftsmanship

一些优化Python代码可读性或执行效率的技巧

## 基本

### comprehension

通常情况下，使用comprehension的写法能让代码可读性更好：

```python title="comprehension.py" linenums="1"
# list comprehension
l = [i for i in range(10) if i % 2 == 0]

# dict comprehension
d = {i:i**2 for i in range(10)}

# set comprehension
s = {i**2 for i in range(10)}
```

### enumerate

同时遍历索引和对象：

```python title="enumerate.py" linenums="1"
for i, element in enumerate(['one', 'two', 'three']):
    print(i, element)
```

### zip

方便地遍历两个list：

```python title="zip1.py" linenums="1"
for i, j in zip([1, 2, 3], [4, 5, 6]):
    print(i, j)
```

或更多list：

```python title="zip2.py" linenums="1"
for i, j, k in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(i, j, k)
```

当list不等长时，遍历完最短的就结束：

```python title="zip.py" linenums="1"
for i, j in zip([1, 2], [4, 5, 6]):
    print(i, j)
```

### unpacking

一些例子：

```python title="unpacking1.py" linenums="1"
a, b, *c = 1, 2, 3, 4
a, *b, c = 1, 2, 3, 4
```

写算法题时候很实用：

```python title="unpacking2.py" linenums="1"
ans = [1, 2, 3]
print(*ans)
```

## 进阶

### string concatenation

由于Python中的str类对象是不可变的(immutable)，因此下面的代码会不断创建新的str类对象，导致复杂度是substring个数的平方：

```python title="inefficient.py" linenums="1"
substrings = ['Some ', 'substrings ', 'to ', 'concatenate.']
s = ''
for substring in substrings:
    s += substring
```

通常情况下，使用join()会使得代码运行的效率更高：

```python title="efficient.py" linenums="1"
substrings = ['Some ', 'substrings ', 'to ', 'concatenate.']
s = ''.join(substrings)
```
