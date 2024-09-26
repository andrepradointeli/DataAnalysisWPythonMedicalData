import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / (df['height']/100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

def draw_cat_plot():
    # Convert the data into long format
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'],
                     var_name='variable', value_name='value')

    # Create the catplot
    g = sns.catplot(x="variable", hue="value", col="cardio",
                    data=df_cat, kind="count",
                    height=5, aspect=1.2)

    # Adjust labels and plot
    g.set_axis_labels("variable", "total")  # Set x-axis label to lowercase 'variable'
    g.set_titles("{col_name}")

    # Save the plot as 'catplot.png'
    fig = g.fig  # Get the figure for saving
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    # Calculate the correlation matrix
    corr = df_heat.corr()

    # 13
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", ax=ax, cmap='coolwarm', cbar_kws={'shrink': .5}, center=0)

    # 16
    # Save the plot as 'heatmap.png'
    fig.savefig('heatmap.png')
    return fig
