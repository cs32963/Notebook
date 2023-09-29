---
date: 2023-05-23
authors:
  - ludwig
categories:
  - 基础
---

# Markdown Syntax in Material for MkDocs

Markdown是一种轻量化的标记语言，非常适合用于编写技术文档。本文展示了用于编写这本Notebook的主要Markdown语法。

<!-- more -->

# 一级标题

## 二级标题

### 三级标题

#### 四级标题

##### 五级标题

###### 六级标题

## 段落

纯文本

另外一些纯文本

## 强调语法

**粗体**

*斜体*

~~删除线~~

## 分隔线

---

## 引用

> Stay hungry. Stay foolish.

## 代码块

代码块支持添加标题、行号，以及高亮部分行

```python title="hello_world.py" linenums="1" hl_lines="2-3"
print('Hello World!')
print('Hello World!')
print('Hello World!')
print('Hello World!')
```

## 数学公式

质能方程$E=mc^2$是一个很有名的公式。

$$
E=mc^2
$$

## 链接

[Markdown官方教程](https://markdown.com.cn)

[Material for MkDocs Reference](https://squidfunk.github.io/mkdocs-material/reference/)

## 文献引用

Transformer[^attention]

Transformer-based Models[^attention][^bert]

## 图片

支持添加图片标题

<figure markdown>
  ![一只可爱的猫猫](./cat.png)
  <figcaption>一只可爱的猫猫</figcaption>
</figure>




[^attention]: Vaswani et al. [Attention Is All You Need.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (NIPS 2017)

[^bert]: Devlin et al. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.](https://aclanthology.org/N19-1423.pdf) (NAACL 2019)
