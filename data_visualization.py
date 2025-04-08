import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Load the Titanic dataset
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df

df = load_data()

# Display the dataset
st.write("Titanic Dataset", df.head())

# Number of survivors vs. non-survivors
st.subheader("Number of Survivors vs. Non-Survivors")
survivor_counts = df['Survived'].value_counts()
st.bar_chart(survivor_counts)

# Passenger class distribution
st.subheader("Passenger Class Distribution")
pclass_counts = df['Pclass'].value_counts().sort_index()
st.bar_chart(pclass_counts)

# Age distribution (drop NaN values first)
st.subheader("Age Distribution")
st.write("Age distribution of passengers")
df_cleaned = df.dropna(subset=['Age'])

# Create a Matplotlib figure and axis explicitly
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_cleaned['Age'], bins=20, color='skyblue', edgecolor='black')
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")

# Display the figure with Streamlit
st.pyplot(fig)

# Gender distribution
st.subheader("Gender Distribution")
st.write("Gender distribution of passengers")

fig, ax = plt.subplots()
df['Sex'].value_counts().plot(kind='bar', ax=ax, color=['lightblue', 'pink'])
ax.set_title("Gender Distribution")
ax.set_xlabel("Gender")
ax.set_ylabel("Count")

st.pyplot(fig)

# Survival by Age and Gender
st.subheader("Survival by Age and Gender")

# Drop rows with missing Age values
df_age_gender = df.dropna(subset=['Age'])

# Map survival for color and sex for symbol
fig_scatter = px.scatter(df_age_gender, x='Age', y='Survived',
                         color='Sex',
                         symbol='Sex',
                         title='Survival by Age and Gender',
                         labels={'Survived': 'Survival'},
                         opacity=0.6)

st.plotly_chart(fig_scatter)

# Passenger Class vs. Survival Rate
st.subheader("Passenger Class vs. Survival Rate")

# Calculate survival rate by passenger class
pclass_survival = df.groupby('Pclass')['Survived'].mean().reset_index()

# Plot using Plotly
fig_pclass = px.bar(pclass_survival, x='Pclass', y='Survived',
                    title='Survival Rate by Passenger Class',
                    labels={'Survived': 'Survival Rate', 'Pclass': 'Passenger Class'},
                    color='Pclass',
                    text='Survived')
fig_pclass.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_pclass.update_layout(yaxis=dict(range=[0,1]))

st.plotly_chart(fig_pclass)

# Gender vs. Survival
st.subheader("Gender vs. Survival Rate")

# Calculate survival rate by gender
gender_survival = df.groupby('Sex')['Survived'].mean().reset_index()

# Plot using Plotly for a nicer look
fig_gender = px.bar(gender_survival, x='Sex', y='Survived', 
                    title='Survival Rate by Gender',
                    labels={'Survived': 'Survival Rate'},
                    color='Sex',
                    text='Survived')
fig_gender.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_gender.update_layout(yaxis=dict(range=[0,1]))

st.plotly_chart(fig_gender)

# Heatmap of Survival Rates by Class and Gender
st.subheader("Heatmap: Survival Rate by Class and Gender")

# Pivot table for heatmap
heatmap_data = df.pivot_table(index='Pclass', columns='Sex', values='Survived', aggfunc='mean')

fig_heatmap, ax = plt.subplots()
im = ax.imshow(heatmap_data, cmap="YlGnBu")

# Show labels
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_xticklabels(heatmap_data.columns)
ax.set_yticklabels(heatmap_data.index)
ax.set_xlabel("Sex")
ax.set_ylabel("Passenger Class")
ax.set_title("Survival Rate")

# Display values on heatmap
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        value = heatmap_data.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

st.pyplot(fig_heatmap)

# Siblings/Spouses vs. survival
st.subheader("Siblings/Spouses vs. Survival")
sibsp_survival = df.groupby('SibSp')['Survived'].mean()
st.bar_chart(sibsp_survival)

# Parents/Children vs. survival
st.subheader("Parents/Children vs. Survival")
parch_survival = df.groupby('Parch')['Survived'].mean()
st.bar_chart(parch_survival)

