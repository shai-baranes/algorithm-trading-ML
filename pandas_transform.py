import pandas as pd

df = pd.DataFrame({
    'team': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
    'points': [30, 22, 19, 14, 14, 11, 20, 28]
})


#   team  points
# 0    A      30
# 1    A      22
# 2    A      19
# 3    A      14
# 4    B      14
# 5    B      11
# 6    B      20
# 7    B      28


# Add a column with the mean() points per team, broadcasted to each row:

df['mean_points'] = df.groupby('team')['points'].transform('mean')
df['mean_points_2'] = df.groupby('team')['points'].transform(lambda x: x.mean()) # equivalent to above


#   team  points  mean_points
# 0    A      30        21.25
# 1    A      22        21.25
# 2    A      19        21.25
# 3    A      14        21.25
# 4    B      14        18.25
# 5    B      11        18.25
# 6    B      20        18.25
# 7    B      28        18.25


# summing all members per group to gain 100%
df['percent_of_points'] = df.groupby('team')['points'].transform(lambda x: x / x.sum())

#   team  points  mean_points  mean_points_2  percent_of_points
# 0    A      30        21.25          21.25           0.352941
# 1    A      22        21.25          21.25           0.258824
# 2    A      19        21.25          21.25           0.223529
# 3    A      14        21.25          21.25           0.164706
# 4    B      14        18.25          18.25           0.191781
# 5    B      11        18.25          18.25           0.150685
# 6    B      20        18.25          18.25           0.273973
# 7    B      28        18.25          18.25           0.383562



# Another good usage for 'transform'
#  Filling Missing 'N/A' Values by Group: 
# df['Salary'] = df.groupby('Neighborhood')['Salary'].transform(lambda x: x.fillna(x.mean()))






df = pd.DataFrame({
    'weight': [0, 2],
    'height': [1, 3]
}, index=['cat', 'dog'])
print(df)

#      weight  height
# cat       0       1
# dog       2       3



df.stack()
# cat  weight    0
#      height    1
# dog  weight    2
#      height    3


df.stack().index
# MultiIndex([('cat', 'weight'),
#             ('cat', 'height'),
#             ('dog', 'weight'),
#             ('dog', 'height')],
#            )



df = pd.DataFrame({
    'weight': [0, 2],
    'height': [1, 3],
    'age'   : [3,4],
}, index=['cat', 'dog'])

df.stack()

# cat  weight    0
#      height    1
#      age       3
# dog  weight    2
#      height    3
#      age    

df.stack().index
# cat  weight    0
#      height    1
#      age       3
# dog  weight    2
#      height    3
#      age    




df = pd.DataFrame({
    'weight': [0, 2, 5],
    'height': [1, 3, 6],
    'age'   : [3,4, 2],
}, index=['cat', 'dog', 'fox'])


