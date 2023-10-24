# Craftsmanship

一些优化Python代码的技巧

## String concatenation

由于Python中的str类是不可变的(immutable)，因此下面的代码会不断创建新的str类对象，导致复杂度是序列个数的平方：

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

