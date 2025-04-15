
# Syntax Notes for Pandas DataFrame

## General Syntax

- **Access Columns**: `.columns`
- **Access Index**: `.index` (contains row numbers or row names, index objects)
- **Shape of DataFrame**: `.shape`
- **Descriptive Statistics**: `.describe()` (includes count, mean, std, min, max, and percentiles)
- **Concise Summary**: `.info()` (includes column names, non-null counts, and data types)
- **Data Types**: `.dtypes` (returns data types of all columns)
- **Values as NumPy Array**: `.values` (returns the DataFrame as a NumPy array)

### Subsetting Data
- **Subset Columns**: `df[['col1', 'col2']]` (returns a DataFrame with selected columns)
- **Subset Rows by Position**: `df[0:4]` (returns rows 0 through 3, excluding 4)
- **Subset Rows by Condition**:
  ```python
  df[(df["col"] > 160) & (df["other_col"] == "value")]
  ```
- **Subset Using `isin()`**: `df[df["col"].isin(["val1", "val2"])]` (filters rows with specified values)
- **Sort Rows**: 
  ```python
  df.sort_values(["col1"], ascending=[False])  # Sorts by col1 in descending order
  ```

## Summary Statistics

### For Individual Rows or Columns
- **Minimum Value**: `.min()`
- **Maximum Value**: `.max()`
- **Custom Summary with `.agg()`**:
  ```python
  def pct30(column):
      return column.quantile(0.3)
  dogs["weight_kg"].agg(pct30)  # Custom function for 30th percentile
  dogs["weight_kg"].agg([min, max, pct30])  # Combine multiple functions
  ```

- **Cumulative Operations**: 
  - `.cumsum()` (cumulative sum)
  - `.cumprod()` (cumulative product)
  - `.cummin()` (cumulative minimum)

- **Dropping Duplicates**:
  ```python
  df.drop_duplicates(subset=["col1", "col2"])
  ```

- **Counting Values**: 
  ```python
  df["col"].value_counts()  # Counts occurrences of each unique value
  ```

### Summaries by Group
- **GroupBy with Single Column**:
  ```python
  dogs.groupby("color")["weight_kg"].mean()
  ```

- **GroupBy with Multiple Columns**:
  ```python
  dogs.groupby(["color", "breed"])["weight_kg"].agg([min, max, sum])
  ```

- **Using NumPy Functions**: Compatible with `.agg()` or `.apply()`, e.g., `np.mean`, `np.median`

## Pivot Tables
- **Basic Pivot**:
  ```python
  dogs.pivot_table(values="weight_kg", index="color", aggfunc=np.mean)
  ```

- **Pivot with Multiple Columns**:
  ```python
  dogs.pivot_table(values="weight_kg", index="color", columns="breed", fill_value=0)
  ```

- **Add Row and Column Totals**: `margins=True`

## Explicit Indexing
- **Set Index**: 
  ```python
  dogs_ind = dogs.set_index("name")  # Sets 'name' column as index
  ```

- **Reset Index**:
  ```python
  dogs_ind.reset_index(drop=True)  # Drops the current index
  ```

- **Simplified Subsetting with Index**:
  ```python
  dogs_ind.loc[["Bella", "Stella"]]  # Subset rows using index labels
  ```

- **Multi-level Index**:
  ```python
  dogs_ind3 = dogs.set_index(["breed", "color"])
  dogs_ind3.loc[[("Labrador", "Brown"), ("Chihuahua", "Tan")]]
  ```

- **Sort Index**:
  ```python
  dogs_ind3.sort_index(level=["color", "breed"], ascending=[True, False])
  ```

### Index Notes
- Index values are treated as data.
- Indexing may violate "tidy data" principles.
- Sorting is required before slicing with indexes.

### Slicing
- **Row Slicing**:
  ```python
  dogs_srt.loc["chow chow":"poodle"]
  ```
- **Column Slicing**:
  ```python
  dogs_srt.loc[:, "name":"heights_in_cm"]
  ```

- **Date Index Slicing**:
  ```python
  dogs.loc["2014":"2016"]  # Filters rows within a date range
  ```

### Position-based Subsetting
- **Using `iloc`**:
  ```python
  dogs.iloc[2:5, 1:4]  # Select rows 2 through 4 and columns 1 through 3
  ```

## Additional Notes

- Use `.groupby()` with `.apply()` for custom group-specific operations.
- Combine `.set_index()` with `.reset_index()` to modify and restore original structure efficiently.
- Use `.pivot_table()` with `fill_value` to handle missing data gracefully.
- Indexes can be combined with hierarchical data structures but may add complexity.
- `.describe()` is a quick way to inspect numerical data but does not include non-numerical columns.

## Working with Pivot Tables (refer chapter 3 for review)

- Pivot tables are just data frames with sorted indices
- `.mean(axis="index")` expected behavious, summary stat for each column
- to get for each row, ie across the columns  `axis = "columns"`
- the columns names that you pass in for index, will be the indices, eg, index = ["country","year"], then each horizontal row will have the country first then city, and if columns = "year", the columns will be years, eg 2010 2011 ...and the value will be the summary stat

## Ch4, Visualizing your data

```python
import matplotlib.pyplot as plt
dog_pack["height_cm"].hist() # x axis will represent the heights, and the y axis the number of dogs in each height range
plt.show()

#  adjust the number of bars using the bins argument
dog_pack["height_cm"].hist(bins = 20)
```

Bar plots can reveal relationships between a categorical variable and a number variable, like breed and weight_kg

```python
avg_weight_by_breed = dog_pack.groupby("breed")["weight_kg"].mean()
avg_weight_by_breed.plot(kind = "bar") # breed on x axis, avg weight on y
# add a title using title = "..."
```

Line plots are great for studying the changes in numerical variables over time

```python
df.plot(x = "date", y = "wegith_kg", kind = "line", rot = 45) 
# rotate x axis labels to make them easier to read
```

```python
dog_pack[dog_pack["sex"] == "F"]["height_cm"].hist(alpha = 0.7)
dog_pack[dog_pack["sex"] == "M"]["height_cm"].hist(alpha = 0.7)
# they overlap each other, make them translucent
plt.legend(["F","M"]) # to tell which color is which
```

### Missing Values

```python
dogs.isna() # boolean for every single value, whether its missing or not
dogs.isna().any() # for each column, if there is at least one missing value, gives True
dogs.isna().sum() # number of missing values per column
dogs.isna().sum().plot(kind = "bar") # plotting the missing values
dogs.dropna() # remove rows that contain missing values
dogs.fillna(0) # fill the Nan with this value
```

### Creating DataFrames

```python
my_dict = {
    key1 : value1,
}
# from a list of dictionaries, built up row by row
list_of_dicts = [
    {
        "name":"Ginger", "breed" : "Daschund", "dob":"2019-03-14"
    },
    {
        "name":"Scout", "breed" : "Dalmation", "dob":"2019-03-24"
    },
]
# each key of every dict correcponds to a column name, and each list item ie a dict is a row 
new_dogs= pd.DataFrame(list_of_dicts)

# from a dictionary of lists, built column by column
dict_of_lists = {
    "name" : ["Ginger","Scount"],
    "breed" : [...]
    ...
}
new_dogs = pd.DataFrame(dict_of_lists)
```

### Reading and Writing CSVs

CSV or Comma Separated Values if desgined for tabular data

```python
df= pd.read_csv("filepath")
```
### misc

```python
df.sort_values(by = "col")
```

snippet on using groupby
```python
schools.groupby("borough").agg(
num_schools = ("school_name", "count"),
average_SAT=("total_SAT","mean"),
std_SAT = ("total_SAT", "std")
).reset_index()

max_std_row = borough_stats.loc[borough_stats["std_SAT"].idxmax()]
largest_std_dev = max_std_row.to_frame().T.round(2)
```