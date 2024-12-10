import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# Function to load dataset and model
def load_data_and_model():
    try:
        df = pd.read_csv('my_data.csv')
    except FileNotFoundError:
        st.error("Dataset 'my_data.csv' not found. Please upload the dataset.")
        df = pd.DataFrame()  # Return empty DataFrame

    try:
        model = joblib.load('total_sales_predictor.pkl')
    except FileNotFoundError:
        st.error("Model file 'total_sales_predictor.pkl' not found.")
        model = None  # Return None if the model is unavailable

    return df, model

# Clean data: Convert sales columns to numeric (in case of any non-numeric values)
def clean_data(df):
    numeric_columns = ['critic_score', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'total_sales']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Display static visualizations
def plot_static(df):
    if df.empty:
        st.warning("No data for visualizations.")
        return

    # 1. Top 10 Games by Total Sales (Bar chart)
    st.write("### 🔝 Top 10 Games by Total Sales")
    top_10_sales = df[['title', 'total_sales']].dropna().sort_values(by='total_sales', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='total_sales', y='title', data=top_10_sales, palette='viridis', ax=ax)
    ax.set_title('Top 10 Games by Total Sales', fontsize=16, weight='bold')
    ax.set_xlabel('Total Sales (in millions)', fontsize=12)
    ax.set_ylabel('Game Title', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
        **Penjelasan:**
        Grafik ini menunjukkan 10 game dengan penjualan total tertinggi. Total penjualan dihitung dalam jutaan kopi.
    """)

    # 2. Sales Distribution by Region (Pie chart)
    st.write("### 🌍 Sales Distribution by Region")
    region_sales = df[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']].sum()

    fig, ax = plt.subplots(figsize=(8, 8))
    region_sales.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3', ax=ax)
    ax.set_title('Sales Distribution by Region')
    ax.set_ylabel('')  # Hide the y-label
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
        **Penjelasan:**
        Diagram ini menunjukkan distribusi penjualan berdasarkan region: Amerika Utara (NA), Jepang (JP), Eropa (PAL), dan lainnya.
    """)

    # 3. Average Critic Score by Genre (Bar chart)
    st.write("### 📝 Average Critic Score by Genre")
    avg_critic_score_by_genre = df.groupby('genre')['critic_score'].mean().dropna().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    avg_critic_score_by_genre.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Average Critic Score by Genre', fontsize=16, weight='bold')
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Average Critic Score', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
        **Penjelasan:**
        Grafik ini menunjukkan rata-rata skor kritik berdasarkan genre game. Semakin tinggi nilai rata-rata, semakin baik penilaian kritik terhadap genre tersebut.
    """)

    # 4. Total Sales by Genre (Bar chart)
    st.write("### 🎮 Total Sales by Genre")
    sales_by_genre = df.groupby('genre')['total_sales'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sales_by_genre.plot(kind='bar', color='lightcoral', ax=ax)
    ax.set_title('Total Sales by Genre', fontsize=16, weight='bold')
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Total Sales (in millions)', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
        **Penjelasan:**
        Grafik ini menunjukkan total penjualan berdasarkan genre game. Penjualan dihitung dalam jutaan kopi dan membantu mengetahui genre yang paling laris di pasaran.
    """)

# Encode categorical features
def encode_categorical_features(input_data, df):
    categorical_columns = ['title', 'console', 'genre', 'publisher', 'developer']
    encoder_map = {}

    for col in categorical_columns:
        encoder = LabelEncoder()
        encoder.fit(df[col])
        encoder_map[col] = encoder
        input_data[col] = encoder.transform(input_data[col])

    return input_data, encoder_map

# Dynamic filters for categorical columns
def dynamic_filters(df):
    st.write("### 🎮 Input Game Features")
    selected_title = st.selectbox("Select Title", sorted(df['title'].unique()))
    filtered_df = df[df['title'] == selected_title]

    selected_console = st.selectbox("Select Console", sorted(filtered_df['console'].unique()))
    selected_genre = st.selectbox("Select Genre", sorted(filtered_df['genre'].unique()))
    selected_publisher = st.selectbox("Select Publisher", sorted(filtered_df['publisher'].unique()))
    selected_developer = st.selectbox("Select Developer", sorted(filtered_df['developer'].unique()))

    return selected_title, selected_console, selected_genre, selected_publisher, selected_developer

# Dynamic sliders for numerical columns
def dynamic_sliders(df):
    st.write("### 📊 Input Sales and Scores")
    critic_score = st.slider("Critic Score", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    na_sales = st.slider("NA Sales (millions)", min_value=0.0, max_value=float(df['na_sales'].max()), step=0.1, value=1.0)
    jp_sales = st.slider("JP Sales (millions)", min_value=0.0, max_value=float(df['jp_sales'].max()), step=0.1, value=1.0)
    pal_sales = st.slider("PAL Sales (millions)", min_value=0.0, max_value=float(df['pal_sales'].max()), step=0.1, value=1.0)
    other_sales = st.slider("Other Sales (millions)", min_value=0.0, max_value=float(df['other_sales'].max()), step=0.1, value=1.0)

    return critic_score, na_sales, jp_sales, pal_sales, other_sales

# Predict total sales using the model
def predict_sales(model, df):
    if model is None:
        st.warning("Model not loaded.")
        return

    # Input data from user
    selected_title, selected_console, selected_genre, selected_publisher, selected_developer = dynamic_filters(df)
    critic_score, na_sales, jp_sales, pal_sales, other_sales = dynamic_sliders(df)

    # Prepare input data
    input_data = pd.DataFrame({
        'title': [selected_title],
        'console': [selected_console],
        'genre': [selected_genre],
        'publisher': [selected_publisher],
        'developer': [selected_developer],
        'critic_score': [critic_score],
        'na_sales': [na_sales],
        'jp_sales': [jp_sales],
        'pal_sales': [pal_sales],
        'other_sales': [other_sales]
    })

    # Encode categorical features
    input_data, encoder_map = encode_categorical_features(input_data, df)

    # Prediction
    if st.button("Predict Total Sales"):
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Total Sales: {prediction[0]:.2f} million copies")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Main Streamlit App
def main():
    st.set_page_config(page_title="🎮 Video Game Sales Dashboard", layout="wide")
    st.title("🎮 Video Game Sales Dashboard")

    # Load data and model
    df, model = load_data_and_model()

    # Clean data
    df = clean_data(df)

    # Display data overview
    st.write("### 📊 Dataset Overview")
    st.dataframe(df)

    # Plot visualizations
    plot_static(df)

    # Predict sales
    st.write("### 🔮 Predict Sales")
    predict_sales(model, df)

if __name__ == "__main__":
    main()
