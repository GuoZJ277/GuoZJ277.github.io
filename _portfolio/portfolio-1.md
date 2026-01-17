---
title: "中老年人抑郁-代谢共病发生风险预测模型的展示与评估"
excerpt: "使用Matplotlib进行探索性数据分析，并通过ROC曲线、PR曲线、混淆矩阵来评估模型性能"
collection: portfolio
date: 2026-01-17
tags: ["预测模型", "随机森林", "医疗数据分析"]

---

## 项目背景
本模型针对医疗场景，基于生理指标（血糖、血脂、肾功能等）、生活习惯及人口统计学特征，构建风险预测模型，帮助医疗机构实现早期风险识别与干预，提升健康管理效率。

## 核心实现
### 1. 数据预处理流水线

# 数值特征：中位数填补
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 类别特征：众数填补+One-Hot编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 特征整合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

---

### 2. 模型训练

# 随机森林模型构建
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ))
])

# 训练与评估
rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)
y_prob = rf_pipeline.predict_proba(X_test)[:,1]


### 3. 模型评估结果

# 混淆矩阵分析
![混淆矩阵](./image/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png)

> **结论：** 模型对存活患者预测准确率较高，但对死亡患者存在一定漏诊。

---

# 性能曲线 (ROC & PR Curve)

| ROC 曲线 | Precision-Recall 曲线 |
| :---: | :---: |
| ![ROC Curve](./image/ROC_curve.png) | ![PR Curve](./image/PR_curve.png) |

> **评估指标：** 模型 AUC 达到 **0.80**。

---

# 特征重要性 (SHAP 可视化)
![SHAP Summary Plot](./image/shap_summary_plot.png)

> **核心发现：** **MS（代谢综合征）** 是模型预测的最关键特征。

---

# 数据分布一致性检查
![训练集-验证集对比](./image/%E8%AE%AD%E7%BB%83%E9%9B%86-%E9%AA%8C%E8%AF%81%E9%9B%86%E5%9F%BA%E7%BA%BF%E7%89%B9%E5%BE%81%E5%AF%B9%E6%AF%94.png)

> **数据质量：** 训练集与测试集特征分布基本一致。

This is an item in your portfolio. It can be have images or nice text. If you name the file .md, it will be parsed as markdown. If you name the file .html, it will be parsed as HTML. 
