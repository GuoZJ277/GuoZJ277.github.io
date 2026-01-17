---
title: "中老年人抑郁-代谢共病发生风险预测模型的展示与评估"
excerpt: "Short description of portfolio item number 1<br/><img src='/images/500x300.png'>"
collection: portfolio
tags: ["预测模型", "随机森林", "医疗数据分析"]
---

## 项目背景
本模型针对医疗场景，基于生理指标（血糖、血脂、肾功能等）、生活习惯及人口统计学特征，构建风险预测模型，帮助医疗机构实现早期风险识别与干预，提升健康管理效率。

## 核心实现
### 1. 数据预处理流水线
```python
# 数值特征：中位数填补
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImuter(strategy='median'))
])

# 类别特征：众数填补+One-Hot编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImuter(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 特征整合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


---

This is an item in your portfolio. It can be have images or nice text. If you name the file .md, it will be parsed as markdown. If you name the file .html, it will be parsed as HTML. 
