import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

def load_data():
    url = "Data_Cleaning_Education.csv"
    df = pd.read_csv(url)
    return df

def perform_clustering(data, n_clusters):
    # Select relevant features for clustering
    features = ['OOSR_Primary_Age_Male', 'OOSR_Primary_Age_Female',
                'OOSR_Lower_Secondary_Age_Male', 'OOSR_Lower_Secondary_Age_Female',
                'OOSR_Upper_Secondary_Age_Male', 'OOSR_Upper_Secondary_Age_Female',
                'Completion_Rate_Primary_Male', 'Completion_Rate_Primary_Female',
                'Completion_Rate_Lower_Secondary_Male', 'Completion_Rate_Lower_Secondary_Female',
                'Completion_Rate_Upper_Secondary_Male', 'Completion_Rate_Upper_Secondary_Female']

    X = data[features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    data['Cluster'] = agglomerative.fit_predict(X_scaled)

    return data

# Main function
def main():
    st.title("Analisis Pengelompokkan Negara-Negara Berdasarkan Karakteristik Pendidikan")

    # Load data
    data = load_data()

    # Display data
    st.subheader("Data Preview")
    st.write(data.head())

    # Education Characteristics Analysis
    st.subheader("Analisis Karakteristik Pendidikan")

    # Create subplot figure
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Distribution of Primary Age Male", 
                                                        "Distribution of Primary Age Female",
                                                        "Completion Rate Primary Male vs Female",
                                                        "Completion Rate Lower Secondary Male vs Female"))

    # Visualize distribution of primary age male and female
    fig.add_trace(go.Histogram(x=data['OOSR_Primary_Age_Male'], name='Primary Age Male'), row=1, col=1)
    fig.add_trace(go.Histogram(x=data['OOSR_Primary_Age_Female'], name='Primary Age Female'), row=1, col=2)

    # Visualize completion rate primary male vs female
    fig.add_trace(go.Box(x=data['Completion_Rate_Primary_Male'], name='Primary Male'), row=2, col=1)
    fig.add_trace(go.Box(x=data['Completion_Rate_Primary_Female'], name='Primary Female'), row=2, col=2)

    # Visualize completion rate lower secondary male vs female
    fig.add_trace(go.Box(x=data['Completion_Rate_Lower_Secondary_Male'], name='Lower Secondary Male'), row=2, col=1)
    fig.add_trace(go.Box(x=data['Completion_Rate_Lower_Secondary_Female'], name='Lower Secondary Female'), row=2, col=2)

    # Update layout
    fig.update_layout(height=800, width=1000, showlegend=True)

    # Show the figure
    st.plotly_chart(fig)

    # Perform clustering
    n_clusters = 3  # Number of clusters
    clustered_data = perform_clustering(data, n_clusters)
    st.success("Clustering completed successfully!")

    # Display clustered data separated by cluster
    st.subheader("Clustered Data Separated by Cluster")

    # Create columns layout
    cols = st.columns(n_clusters)
    
    for cluster_id, col in zip(clustered_data['Cluster'].unique(), cols):
        with col:
            st.subheader(f"Cluster {cluster_id}")
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster_id]
            st.write(cluster_data[['Countries and areas']])

    # Explanation of clusters
    st.write("Cluster 0: Negara-negara dalam klaster ini mungkin memiliki karakteristik pendidikan yang lebih baik atau lebih maju. Ini bisa ditunjukkan dengan tingkat penyelesaian yang tinggi di semua tingkatan pendidikan dan jumlah minimal atau tidak ada anak-anak di luar sekolah pada semua tingkatan. Negara-negara dalam klaster ini mungkin memiliki sistem pendidikan yang efektif dan tersedia untuk semua jenis kelamin.")
    st.write("Cluster 1: Klaster ini mungkin mencakup negara-negara dengan tingkat pendidikan yang sedang atau rendah. Ini bisa ditunjukkan dengan jumlah yang signifikan dari anak-anak di luar sekolah di semua tingkatan dan tingkat penyelesaian pendidikan yang rendah. Kemungkinan besar, negara-negara dalam klaster ini membutuhkan perhatian khusus dan sumber daya tambahan untuk meningkatkan akses dan kualitas pendidikan.")
    st.write("Cluster 2: Negara-negara dalam klaster ini mungkin menunjukkan pola pendidikan yang beragam, bervariasi atau menengah. Beberapa negara mungkin memiliki tingkat penyelesaian yang tinggi di tingkat tertentu, sementara yang lain memiliki tantangan di tingkat lain. Klaster ini mungkin mencerminkan keragaman situasi pendidikan di negara-negara tersebut, dan strategi intervensi yang berbeda mungkin diperlukan tergantung pada kebutuhan pendidikan masing-masing negara.")

if __name__ == "__main__":
    main()
