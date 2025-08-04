# Statistical Analysis

### Types

- **Descriptive Statistics** : summarize and describe data characteristics
- **inferential statistics** : draw comclusions about populations from samples
- **predictive statistics** : use current and historical data to predict future outcomes

### Parametric vs Non-parametric tests

- **Parametric tests** : assume data follows a certain distribution (e.g., normal distribution)
- more powerful when assumptions are met, eg t-test when data is normally distributed, anova, pearson correlation
- eg. compariing average cholestrol levels between two groups
- **Non-parametric tests** : do not assume a specific distribution, used when data does
- often use ranks rather than actual values.
- more robust to outliers and skewed data, eg. mann-whitney u test, kruskal-wallis test, spearman correlation
- eg comparing customer satisfaction ratings between different store locations. Ratings usually not normally distributed

### Choosing the right test

| Number of Populations | Data Type      | Appropriate Tests                       |
| --------------------- | -------------- | --------------------------------------- |
| **Two Populations**   | Parametric     | - Independent t-test (unrelated groups) |
|                       |                | - Paired t-test (related groups)        |
|                       |                | - F-test (compare variances)            |
|                       | Non-parametric | - Mann-Whitney U test                   |
|                       |                | - Wilcoxon signed-rank test             |
|                       |                | - Kolmogorov-Smirnov test               |
| **Three or More**     | Parametric     | - One-way ANOVA                         |
|                       |                | - Repeated measures ANOVA               |
|                       |                | - Factorial ANOVA                       |
|                       | Non-parametric | - Kruskal-Wallis test                   |
|                       |                | - Friedman test                         |
|                       |                | - Dunn's test (post-hoc)                |
