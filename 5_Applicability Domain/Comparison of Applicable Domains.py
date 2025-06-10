#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def load_and_preprocess_data():

    aug_all = pd.read_excel(
        r"C:\Users\HP\Desktop\NKU\O3-based AOPs\peroxone.xlsx",
        sheet_name='RefData',
        index_col=0,
        header=0
    )
    model2_data = aug_all[["lg(O3/DOC)", "lg(H2O2/O3)", "pH", "TOC0(mg/L)"]]


    combined_data = pd.read_excel(
        r"C:\Users\HP\jupyternotebook\MLofCC\combined_data.xlsx",
        header=0
    )
    model1_data = combined_data[['lg(O3)', 'lg(H2O2)', 'pH']]
    
    return model1_data, model2_data

def convert_model2_features(model2_data):
    TOC = model2_data["TOC0(mg/L)"].values
    lg_TOC = np.log10(TOC)
    
    lg_O3 = model2_data["lg(O3/DOC)"] + lg_TOC  # lg(O3) = lg(O3/DOC) + lg(TOC)
    lg_H2O2 = model2_data["lg(H2O2/O3)"] + lg_O3  # lg(H2O2) = lg(H2O2/O3) + lg(O3)
    
    return pd.DataFrame({
        'lg(O3)': lg_O3,
        'lg(H2O2)': lg_H2O2,
        'pH': model2_data['pH']
    })

def normalize_data(model1_data, model2_converted):
    scaler = MinMaxScaler()
    scaler.fit(model1_data)  
    
    model1_normalized = scaler.transform(model1_data)
    model2_normalized = scaler.transform(model2_converted)
    
    return model1_normalized, model2_normalized

def visualize_applicability_domains(model1_pca, model2_pca):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(model1_pca[:, 0], model1_pca[:, 1], c='blue', alpha=0.6, label='Model1 Data')
    plt.scatter(model2_pca[:, 0], model2_pca[:, 1], c='red', alpha=0.6, label='Model2 Data')
    
    for data, color in zip([model1_pca, model2_pca], ['blue', 'red']):
        hull = ConvexHull(data)
        for simplex in hull.simplices:
            plt.plot(data[simplex, 0], data[simplex, 1], color=color, linestyle='--', linewidth=1.5)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Applicability Domain Comparison (PCA + Convex Hull)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Applicability_Domain_Comparison.png', dpi=300)
    plt.show()

def calculate_metrics(model1_pca, model2_pca):

    hull1 = ConvexHull(model1_pca)
    hull2 = ConvexHull(model2_pca)
    
    poly1 = Polygon(model1_pca[hull1.vertices])
    poly2 = Polygon(model2_pca[hull2.vertices])
    
    overlap = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = overlap / union if union != 0 else 0
    
    hausdorff = max(
        directed_hausdorff(model1_pca, model2_pca)[0],
        directed_hausdorff(model2_pca, model1_pca)[0]
    )
    
    return iou, hausdorff

if __name__ == "__main__":
    
    model1_data, model2_raw = load_and_preprocess_data()
    

    model2_converted = convert_model2_features(model2_raw)
    

    model1_norm, model2_norm = normalize_data(model1_data, model2_converted)
    

    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(np.vstack([model1_norm, model2_norm]))
    model1_pca = combined_pca[:len(model1_norm)]
    model2_pca = combined_pca[len(model1_norm):]
    

    visualize_applicability_domains(model1_pca, model2_pca)
    

    iou, hausdorff = calculate_metrics(model1_pca, model2_pca)

