# 电商用户流失风险预测与客户留存分析系统
### From Predictive Analytics to Proactive Customer Retention

**SQL · MySQL · Python · Pandas · Scikit-learn · Random Forest · Power BI**

[English README](./README.md) | 中文项目介绍

---

## 一、项目概述（Executive Summary）

本项目旨在构建一套基于用户行为数据的电商用户流失风险预测与主动留存分析系统。

传统用户流失分析通常依赖历史数据识别已经停止活跃的用户，而本项目采用基于时间窗口的行为特征工程方法，利用用户近期的浏览、收藏、加购及购买行为，识别具有潜在流失风险的用户，并为不同用户群体设计差异化留存策略。

项目结合 MySQL、SQL、Python 和 Power BI，构建从用户行为数据处理、特征工程、风险预测、模型评估到留存策略分析的完整数据分析流程。

**核心项目成果：**

| 分析指标 | 项目结果 |
|---|---:|
| 建模用户数量 | 654 |
| 流失风险用户数量 | 357 |
| 流失风险用户占比 | 54.59% |
| 特征生成窗口 | 8天 |
| 后续观察窗口 | 1天 |
| 最终模型 | Random Forest |
| 最终分类阈值 | 0.40 |
| 模型 Recall | 94.4% |
| Control Group 7-day Retention | 21.5% |
| Treatment Group 7-day Retention | 28.8% |
| 留存率绝对提升 | 7.3个百分点 |
| A/B Testing P-value | 0.00016553 |

本项目不仅关注预测模型的分类表现，也关注模型结果如何转化为可执行的客户留存策略。

---

## 二、业务背景与分析目标（Business Context & Objectives）

在电商平台中，获取新用户通常需要投入营销和获客成本，因此，提高现有用户的留存率是重要的业务目标。

传统描述性分析可以识别已经停止活跃的用户，但往往难以在用户完全流失之前提供足够及时的干预信号。

本项目希望通过用户近期行为数据建立一个主动预警流程，在用户购买意向减弱或活跃行为发生变化时，识别潜在高风险用户，并为后续营销和客户运营提供数据支持。

### 核心业务问题

1. 如何根据用户近期行为识别潜在流失风险？
2. 哪些行为特征对流失风险预测更有价值？
3. 如何通过机器学习模型提高高风险用户识别能力？
4. 不同价值及风险等级的用户应该采用什么留存策略？
5. 优惠券等干预措施是否能够改善短期用户留存表现？

### 项目目标

- 使用 SQL 构建用户级行为特征数据集。
- 通过滑动时间窗口区分特征生成期和风险观察期。
- 使用 Logistic Regression 和 Random Forest 建立预测模型。
- 通过 Recall 和分类阈值优化提高高风险用户识别能力。
- 分析关键行为特征的重要性。
- 根据用户价值及流失风险设计差异化留存策略。
- 使用 A/B Testing 分析框架评估留存干预效果。
- 通过 Power BI 构建用户风险监控和留存分析 Dashboard。

---

## 三、数据定义与特征工程（Data Definition & Feature Engineering）

### 3.1 流失风险定义（Churn-risk Definition）

本项目采用基于未来短期行为的风险标签定义。

如果用户在特征生成窗口结束后的未来1天观察期内，没有出现以下任一行为：

- Cart：加入购物车
- Buy：购买

则将该用户标记为：

`is_churn = 1`

否则标记为：

`is_churn = 0`

需要注意，这里的 `is_churn` 表示项目定义的短期行为风险标签，并不等同于已经验证的长期客户流失。

### 3.2 滑动窗口设计（Sliding-window Design）

项目使用两个连续的时间窗口：

| 时间窗口 | 长度 | 用途 |
|---|---:|---|
| Feature Generation Window | 8天 | 提取用户历史行为特征 |
| Churn Observation Window | 1天 | 判断用户后续是否出现 Cart 或 Buy 行为 |

选择8天特征窗口，是为了避免更短的观察窗口产生过于稀疏的用户行为特征。

### 3.3 核心行为特征（Behavioral Features）

| 特征 | 含义 | 业务解释 |
|---|---|---|
| recency_hours | 最近一次用户交互距离特征窗口结束的小时数 | 衡量用户近期活跃程度 |
| pv_count_8d | 8天内浏览次数 | 衡量用户浏览活跃度 |
| fav_count_8d | 8天内收藏次数 | 衡量用户商品兴趣 |
| cart_count_8d | 8天内加购次数 | 衡量用户购买意向 |
| buy_count_8d | 8天内购买次数 | 衡量用户实际购买行为 |
| pv_to_buy_ratio | 浏览与购买次数之间的比率 | 反映浏览到购买的行为关系 |

### 3.4 SQL 特征工程

使用 MySQL 构建用户级特征表。

主要技术包括：

- Common Table Expressions（CTEs）
- Conditional Aggregation
- Time-window Filtering
- User-level Feature Aggregation
- Derived Feature Construction

通过 SQL 将原始用户行为日志转换为结构化用户特征表，为后续模型训练提供数据基础。

---

## 四、SQL 数据处理流程（SQL Feature Engineering Pipeline）

核心 SQL 文件：

`user_churn_etl.sql`

### 数据处理流程

1. 从原始用户行为数据中提取用户级记录。
2. 根据时间边界划分8天特征生成窗口和后续1天观察窗口。
3. 使用条件聚合统计不同用户行为次数。
4. 计算最近交互时间及浏览购买比率。
5. 根据未来观察窗口生成短期流失风险标签。
6. 过滤没有有效浏览或加购记录的用户。
7. 生成用户级结构化特征表。

### 数据集构建结果

最终提取654名具有有效活动记录的用户，其中：

- 357名用户被标记为短期流失风险用户。
- 其余297名用户未被标记为短期流失风险用户。

这些数据用于后续分类模型训练及测试。

---

## 五、流失风险预测模型（Churn Prediction Modeling）

### 5.1 模型选择

本项目首先使用 Logistic Regression 作为基准分类模型，随后采用 Random Forest Classifier 进行最终模型训练。

| 模型 | 主要作用 |
|---|---|
| Logistic Regression | 建立基准模型，提供线性分类参考 |
| Random Forest | 捕捉用户行为特征之间的非线性关系 |

经过模型比较，Random Forest 在项目中表现出更好的 Recall，因此被选为最终模型。

### 5.2 模型优化（Model Optimization）

项目采用分类阈值调整策略。

默认分类阈值：

`0.50`

最终采用：

`0.40`

调整阈值的业务考虑是：

漏掉潜在高风险用户可能造成更高的留存机会损失，而对部分非高风险用户进行额外营销触达则可能产生一定优惠券成本。

因此，项目优先关注模型识别高风险用户的能力。

### 5.3 模型评估结果

| 评估指标 | 最终模型 |
|---|---:|
| Model | Random Forest |
| Classification Threshold | 0.40 |
| Recall | 94.4% |

**核心结果：**

最终模型在测试数据中取得94.4%的 Recall，能够识别大部分被标记为短期流失风险的用户。

这一指标反映的是模型对正类风险用户的识别能力，不应直接解释为整体预测准确率。

### 5.4 特征重要性分析（Feature Importance）

项目的特征重要性分析发现：

`cart_count_8d` 是模型中较强的预测特征之一。

这一结果说明，加购行为相较于简单浏览行为，可能包含更明显的购买意向信息。

如果用户此前表现出较强的购买意向，但近期相关行为减少，则可能值得进一步关注。

**业务意义：**

用户风险预测不应只关注浏览次数，还应结合加购、收藏及购买等不同深度的行为信号。

---

## 六、客户留存策略设计（Retention Strategy Analysis）

在识别潜在高风险用户之后，项目进一步根据用户历史消费价值和流失风险制定差异化留存方案。

### 6.1 用户分群与留存策略

| 用户群体 | 风险特征 | 建议留存策略 |
|---|---|---|
| High-Value / High-Risk | 历史购买价值较高，当前流失风险较高 | 优惠券、个性化通知及重点客户召回 |
| Medium-Value / High-Risk | 有一定购买活动，但近期活跃程度下降 | 相关商品推荐、品类促销 |
| Low-Value / High-Risk | 历史消费和用户活动较低 | 低成本营销触达、会员积分及签到活动 |

### 6.2 策略设计逻辑

不同用户群体的历史价值和风险程度不同，因此不应该采用完全相同的留存策略。

对于高价值用户，应优先考虑召回和维护。

对于中等价值用户，可以通过商品推荐及营销活动促进再次购买。

对于低价值用户，应优先采用成本较低的触达方式，避免过度投入营销资源。

---

## 七、商业影响估算（Business Impact Estimation）

为了评估主动留存策略的潜在商业价值，项目建立了简化的 ROI 情景估算模型。

### 7.1 估算公式

\[
\Delta GMV =
N_{target}
\times Recall
\times ConversionRate
\times AOV
-
Cost_{coupons}
\]

其中：

| 参数 | 含义 |
|---|---|
| N_target | 目标高风险用户数量 |
| Recall | 模型对风险用户的识别率 |
| ConversionRate | 预估干预后重新激活的转化率 |
| AOV | 平均订单金额 |
| Cost_coupons | 优惠券成本 |

### 7.2 模拟假设

项目采用以下假设：

| 参数 | 假设值 |
|---|---:|
| 目标高风险用户 | 10,000 |
| Model Recall | 94.4% |
| Estimated Reactivation Conversion Rate | 10% |
| Average Order Value | RMB 150 |

### 7.3 业务解读

该模型用于估算在更大规模用户群体中实施主动留存策略的潜在商业价值。

需要注意：

以上属于简化的业务情景模拟，不是实际生产环境中的收入预测，也不能直接视为已经实现的 GMV 增长。

实际商业收益还需要考虑干预成本、优惠券核销、自然回流用户及真实增量转化等因素。

---

## 八、Cohort Analysis 与 A/B Testing（Retention Experimentation）

为了进一步评估留存干预策略的效果，项目使用 A/B Testing 分析框架比较不同实验组的短期用户留存表现。

### 8.1 实验分组

| 实验组 | 干预方式 |
|---|---|
| Control Group | 不接受留存激励 |
| Treatment Group | 接受定向优惠券干预 |

核心分析文件：

`retention_ab_test_analyzer.ipynb`

### 8.2 7-Day Retention Results

| 指标 | Control Group | Treatment Group |
|---|---:|---:|
| 7-Day Retention Rate | 21.5% | 28.8% |

Treatment Group 的7天留存率高于 Control Group。

### 8.3 绝对提升（Absolute Uplift）

\[
28.8\%-21.5\%=7.3
\]

即：

**7.3个百分点（percentage points）**

### 8.4 相对提升（Relative Uplift）

\[
\frac{28.8\%-21.5\%}{21.5\%}
\times100\%
\approx33.95\%
\]

即相对留存率提升约33.95%。

---

## 九、统计检验与实验结果验证（Statistical Validation）

### 9.1 假设检验

项目使用 Independent Two-sample T-test 对两个实验组的留存表现进行比较。

**Null Hypothesis (H₀)**

优惠券留存干预不会造成两组短期留存表现的差异。

**Alternative Hypothesis (H₁)**

两组短期留存表现存在差异。

### 9.2 检验结果

| 统计指标 | 结果 |
|---|---:|
| T-Statistic | -3.7737 |
| P-Value | 0.00016553 |
| Significance Level | 0.01 |

由于：

\[
p<0.01
\]

因此，在项目所采用的统计检验框架下，拒绝原假设。

### 9.3 结果解读

项目记录显示，Treatment Group 的7天留存率高于 Control Group，且两组差异达到统计显著性水平。

这一结果支持进一步研究定向留存干预策略的价值。

需要说明的是，项目的实验输出记录包含模拟实验数据汇总，因此上述结果应理解为项目实验分析框架中的结果，不应直接当作真实线上随机对照实验的因果结论。

---



## 十、Dashboard 展示（Dashboard Showcase）

本项目包含四个 Power BI Dashboard 页面，分别展示整体业务表现、用户留存、用户风险分群及购买转化分析。

### Page 1 — Executive Overview

![Executive Overview](./assets/Executive_Overview.png)

整体展示用户数量、流失率、平均订单金额及核心业务指标。

### Page 2 — User Segmentation

![User Segmentation](./assets/User_Segmentation.png)

展示不同用户群体的风险分布、行为特征及高风险用户识别结果。

### Page 3 — Retention Heatmap

![Retention Heatmap](./assets/Retention_Heatmap.png)

展示 Cohort Retention、用户留存热力图及不同用户群体的留存表现。

### Page 4 — Funnel Optimization

![Funnel Optimization](./assets/Funnel_Optimization.png)

展示 Page View、Add to Cart、Purchase 及购买转化路径。

---


## 十一、项目目录结构（Repository Structure）

```text
Customer-Churn-Prediction-Retention-Analytics/
│
├── README.md
│
├── user_churn_etl.sql
│
├── Churn_Prediction_Framework.ipynb
│
├── retention_ab_test_analyzer.ipynb
│
├── Result_Data/
│   └── Simulated Experiment Output
│
├── dashboards/
│   ├── Executive_Overview.pbix
│   ├── Funnel_Optimization.pbix
│   ├── Retention_Heatmap.pbix
│   └── User_Segmentation.pbix
│
└── assets/
    ├── Page1_Executive_Overview.png
    ├── Page2_User_Segmentation.png
    ├── Page3_Retention_Heatmap.png
    └── Page4_Funnel_Optimization.png
```

---

## 十二、技术栈（Technology Stack）

| 技术类别 | 使用工具 | 主要用途 |
|---|---|---|
| 数据查询 | SQL / MySQL | 用户行为数据提取与特征工程 |
| 数据处理 | Python / Pandas | 数据处理与特征准备 |
| 机器学习 | Scikit-learn | Logistic Regression、Random Forest |
| 模型评估 | Recall、Threshold Optimization | 风险用户识别能力优化 |
| 统计分析 | Statistical Testing | 留存实验结果验证 |
| 商业智能 | Power BI | 用户分群、风险监控与留存分析 |
| 项目开发 | Jupyter Notebook | 建模与实验分析 |

---

## 十三、项目能力展示（Skills Demonstrated）

**SQL & Data Engineering**
- MySQL
- CTEs
- Conditional Aggregation
- Sliding-window Feature Engineering
- User-level Feature Construction

**Machine Learning**
- Logistic Regression
- Random Forest Classification
- Recall Optimization
- Classification Threshold Tuning
- Feature Importance Analysis

**Business Analytics**
- Customer Churn Risk Identification
- Customer Segmentation
- Retention Strategy Design
- ROI Simulation
- Business Impact Estimation

**Experimentation & BI**
- Cohort Analysis
- A/B Testing
- Independent Two-sample T-test
- Power BI Dashboard Development

---

## 十四、项目范围与局限性（Scope & Limitations）

- **风险标签定义：** 本项目将未来1天内没有 Cart 或 Buy 行为的用户定义为短期流失风险用户，不等同于长期客户流失。
- **特征窗口：** 使用8天行为数据构建用户特征。
- **建模样本：** 最终提取654名有效活动用户。
- **模型指标：** 94.4%为测试集 Recall，不代表 Accuracy。
- **阈值选择：** 使用0.40作为最终分类阈值，以提高风险用户识别能力。
- **实验结果：** A/B Testing 输出包含模拟实验数据汇总，不应解释为已经完成的真实线上随机实验。
- **商业影响：** ROI 和 GMV 计算属于情景估算，实际收益需要进一步验证。
- **统计解释：** 统计显著性不能脱离实验设计、样本独立性和数据生成过程单独解释。

本项目是独立的电商数据分析与机器学习作品集项目，重点展示基于 SQL 的用户行为特征工程、流失风险预测、留存策略设计及商业决策支持能力。

---

**Built with SQL, MySQL, Python, Scikit-learn and Power BI.**
