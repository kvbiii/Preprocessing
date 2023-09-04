from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
from sklearn.cluster import KMeans

class Other_Plots():
    def __init__(self):
        pass

    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        data = np.array(data)
        if(data.ndim == 2):
            data = data.squeeze()
        return data
    
    def check_2d_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        return np.array(data)

    def cross_validation_split(self, X, y, n_splits, cv, n_repeats=1):
        fig = go.Figure()
        if(len(X) > 100):
            symbol = "line-ns"
            marker_size = min(100/n_splits, 10)
            marker_width = min(50/n_splits, 2)
        else:
            symbol = "hexagon"
            marker_width = 10
            marker_size = 10
        n_repeats = 0 
        split = 0
        for split, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            if(split == 0):
                showlegend = True
            else:
                showlegend = False
            fig.add_trace(go.Scatter(x=train_idx, y=[split+1 for i in range(len(train_idx))], mode='markers', marker_symbol=symbol, marker_color="blue", marker_line_color="blue", marker_line_width=marker_width, marker_size=marker_size, showlegend=showlegend, name="Train"))
            fig.add_trace(go.Scatter(x=test_idx, y=[split+1 for i in range(len(test_idx))], mode='markers', marker_symbol=symbol, marker_color="red", marker_line_color="red", marker_line_width=marker_width, marker_size=marker_size, showlegend=showlegend, name="Test"))
            if((split+1)%n_splits==0):
                n_repeats = n_repeats+1
        fig.update_layout(template="simple_white",  xaxis_title="Indices", width=1000, height=1000, font=dict(family="Times New Roman",size=16,color="Black"),  yaxis=dict(ticks="outside", tickvals=[i for i in range(1, n_repeats*n_splits+1)], ticktext=["Fold {}".format(n_splits if i%n_splits==0 else i%n_splits) for i in range(1, n_repeats*n_splits+1)]))
        fig.show("png")
    
    def barplot_anova(self, X, y, feature_name):
        X = self.check_data(data=X)
        y = self.check_data(data=y)
        fig = go.Figure()
        for category in np.unique(X):
            y_subset = y[X==category]
            fig.add_trace(go.Box(y=y_subset, name=category))
        fig.update_layout(template="simple_white", width=600, xaxis_title=feature_name, height=600, showlegend=False, title=f"<b>Comparision of values for {feature_name} categories<b>", title_x=0.5, yaxis_title="Data values", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def scatter_plot_of_ranks(self, spearman_ranks, hoeffding_ranks, quantile=0.75):
        x, y = [], []
        for feature in hoeffding_ranks.keys():
            x.append(hoeffding_ranks[feature])
            y.append(spearman_ranks[feature])
        low_ranks_features = [feature for feature in hoeffding_ranks.keys() if (hoeffding_ranks[feature] >= np.quantile(x, q=quantile)) and (spearman_ranks[feature] >= np.quantile(y, q=quantile))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color="white", size=10, line=dict(color="black", width=1)), name=""))
        fig.add_vline(x=np.quantile(x, q=quantile), line_dash="dash", line_color="red", line_width=2)
        fig.add_hline(y=np.quantile(y, q=quantile), line_dash="dash", line_color="red", line_width=2)
        for feature in low_ranks_features:
            fig.add_annotation(x=hoeffding_ranks[feature], y=spearman_ranks[feature], text=feature, showarrow=False, arrowhead=1, font=dict(family="Times New Roman",size=16,color="Black"), yshift=15)
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Scatter Plot of the ranks<b>", title_x=0.5, xaxis_title="Hoeffding Ranks", yaxis_title="Spearman Ranks", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")

    def cumulative_explained_variance(self, fitted_pca):
        labels = [i+1 for i in range(0, fitted_pca.components_.shape[1])]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=fitted_pca.explained_variance_ratio_, marker=dict(line=dict(color='black', width=1))))
        fig.add_trace(go.Scatter(x=labels, y=np.cumsum(fitted_pca.explained_variance_ratio_), marker=dict(line=dict(color='black', width=1)), line=dict(color="black")))
        fig.update_layout(template="simple_white", width=max(30*len(labels), 600), height=max(30*len(labels), 600), title=f"<b>Cumulative explained variance<b>", title_x=0.5, yaxis_title="Percentage of explained vairance", xaxis=dict(title='Principle Component', showticklabels=True, type="category"), font=dict(family="Times New Roman",size=16,color="Black"), showlegend=False)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", line_width=3)
        fig.show("png")

    def eigenvectors_2d(self, fitted_pca, feature1, feature2):
        feature1 = self.check_data(data=feature1)
        feature2 = self.check_data(data=feature2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feature1, y=feature2, mode='markers', line=dict(color="lightblue"), name="Real values"))
        for component in range(0, fitted_pca.components_.shape[1]):
            #Multiplying by three below for length is just a scalar to better visualize the length and variance explained by vectors
            fig.add_annotation(ax=fitted_pca.mean_[0], ay=fitted_pca.mean_[1], axref="x", ayref="y", x=fitted_pca.components_[0, component]*np.sqrt(fitted_pca.explained_variance_[component])*3, y=fitted_pca.components_[1, component]*np.sqrt(fitted_pca.explained_variance_[component])*3, showarrow=True, arrowsize=1, arrowhead=2, arrowwidth=3, xanchor="right", yanchor="top")
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Eigenvectors<b>", title_x=0.5, xaxis_title="Feature 1", yaxis_title="Feature 2", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def iversed_plot(self, fitted_pca, feature1, feature2, feature1_inversed, feature2_inversed):
        feature1 = self.check_data(data=feature1)
        feature2 = self.check_data(data=feature2)
        feature1_inversed = self.check_data(data=feature1_inversed)
        feature2_inversed = self.check_data(data=feature2_inversed)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feature1, y=feature2, mode='markers', line=dict(color="lightblue"), name="Real Values"))
        fig.add_trace(go.Scatter(x=feature1_inversed, y=feature2_inversed, mode='markers', line=dict(color="blue"), name="Projected"))
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Inverse Transform<b><br>Ratio of explained variance: {}".format(np.round(fitted_pca.explained_variance_ratio_[0], 4)), title_x=0.5, xaxis_title="Feature 1", yaxis_title="Feature 2", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def dendrogram_plot(self, children, distances, labels, truncate_mode="level", p=5, hline_level=None):
        plt.figure(figsize=(10,6))
        plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
        plt.ylabel('Distance', fontsize=12)
        counts = np.zeros(children.shape[0])
        n_samples = len(labels)
        for i, merge in enumerate(children):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack([children, distances, counts]).astype(float)
        dendrogram(linkage_matrix, truncate_mode=truncate_mode, p=p, labels=labels, leaf_rotation=80, leaf_font_size=12, show_contracted=True)
        if(hline_level != None):
            plt.axhline(y=hline_level, color='r', linestyle='--')
        plt.show()
    
    def elbow_plot(self, data, algorithm_instance, max_clusters=15):
        data = self.check_2d_data(data=data)
        sse = []
        for k in range(1, max_clusters+1):
            algorithm_instance.set_params(n_clusters=k)
            algorithm_instance.fit(data)
            sse_data = 0
            for cluster in np.unique(algorithm_instance.labels_):
                X_cluster = data[np.where(algorithm_instance.labels_ == cluster)]
                center_of_cluster = np.mean(X_cluster, axis=0)
                sse_data += np.linalg.norm(X_cluster-center_of_cluster)**2
            sse.append(sse_data)
        for k in range(0, len(sse)):
            if(k > 0 and k < len(sse)-1):
                if(sse[k-1]-sse[k+1] < 0.05*np.sum(sse)):
                    optimal_k = k
                    optimal_sse = sse[k]
                    break
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[k for k in range(1, max_clusters+1)], y=sse, mode='lines+markers'))
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Number of clusters (k)", yaxis_title="Distortion Score", showlegend=False, title=f"<b>Elbow Method<b><br>Optimal SSE: {np.round(optimal_sse, 4)} for k={optimal_k}", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", line_width=2)
        fig.show("png")
    
    def silhouette_plot(self, data, algorithm_instance, max_clusters=10):
        data = self.check_2d_data(data=data)
        best_silhouette_score = -np.inf
        silhouette_scores = []
        for k in range(2, max_clusters+1):
            algorithm_instance.set_params(n_clusters=k)
            algorithm_instance.fit(data)
            silhouette_scores.append(np.mean(self.calculate_silhouette_coefficient(data=data, algorithm_instance=algorithm_instance)))
            if(silhouette_scores[k-2] > best_silhouette_score):
                best_silhouette_score = silhouette_scores[k-2]
                optimal_k = k
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[k for k in range(2, max_clusters+1)], y=silhouette_scores, mode='lines+markers'))
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Number of clusters (k)", yaxis_title="Distortion Score", showlegend=False, title=f"<b>Silhouette Method<b><br>Optimal Silhouette Coefficient: {np.round(best_silhouette_score, 4)} for k={optimal_k}", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", line_width=2)
        fig.show("png")
    
    def calculate_silhouette_coefficient(self, data, algorithm_instance):
        algorithm_instance.fit(data)
        silhouette_coefficients = []
        for cluster in np.unique(algorithm_instance.labels_):
            X_cluster = data[np.where(algorithm_instance.labels_ == cluster)]
            if(X_cluster.shape[0] == 1):
                silhouette_coefficients.append([0])
                continue
            a_i = 1/(X_cluster.shape[0]-1)*np.sum(np.sqrt(np.sum((X_cluster[:,np.newaxis]-X_cluster)**2, axis=2)), axis=1)
            min_bi = np.array([np.inf for i in range(X_cluster.shape[0])])
            for other_cluster in np.unique(algorithm_instance.labels_):
                if(other_cluster != cluster):
                    X_other_cluster = data[np.where(algorithm_instance.labels_ == other_cluster)]
                    b_i = 1/(X_other_cluster.shape[0])*np.sum(np.sqrt(np.sum((X_cluster[:,np.newaxis]-X_other_cluster)**2, axis=2)), axis=1)
                    min_bi = np.min([min_bi, b_i], axis=0)
            silhouette_coefficients.append(list((min_bi-a_i)/np.max([a_i, min_bi], axis=0)))
        silhouette_coefficients = [item for sublist in silhouette_coefficients for item in sublist]
        return silhouette_coefficients
    
    def gap_plot(self, data, algorithm_instance, number_of_reference_datasets=30, max_clusters=15):
        data = self.check_2d_data(data=data)
        gaps = []
        stds = []
        optimal_k = None
        ranges = np.array([[np.min(data[:, feature]), np.max(data[:, feature])] for feature in range(0, data.shape[1])])
        for k in range(1, max_clusters+1):
            sse_references = []
            for reference in range(number_of_reference_datasets):
                reference_data = np.random.uniform(low=ranges[:, 0], high=ranges[:, 1], size=(data.shape[0], data.shape[1]))
                algorithm_instance.set_params(n_clusters=k)
                algorithm_instance.fit(reference_data)
                sse = 0
                for cluster in np.unique(algorithm_instance.labels_):
                    X_cluster = reference_data[np.where(algorithm_instance.labels_ == cluster)]
                    center_of_cluster = np.mean(X_cluster, axis=0)
                    sse += np.linalg.norm(X_cluster-center_of_cluster)**2
                sse_references.append(sse)
            algorithm_instance.set_params(n_clusters=k)
            algorithm_instance.fit(data)
            sse_data = 0
            for cluster in np.unique(algorithm_instance.labels_):
                X_cluster = data[np.where(algorithm_instance.labels_ == cluster)]
                center_of_cluster = np.mean(X_cluster, axis=0)
                sse_data += np.linalg.norm(X_cluster-center_of_cluster)**2
            gap = np.log(np.mean(sse_references)) - np.log(sse_data)
            gaps.append(gap)
            std = np.std(np.log(sse_references))
            stds.append(std)
            if(k > 1 and optimal_k == None):
                if(gaps[k-2] >= gaps[k-1] - stds[k-1]):
                    optimal_k = k-1
                    maximum_gap = gaps[k-2]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[k for k in range(1, max_clusters+1)], y=gaps, mode='lines+markers', error_y=dict(type='data', array=stds)))
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Number of clusters (k)", yaxis_title="Gap statistic (k)", showlegend=False, title=f"<b>Gap statistic<b><br>Maximum gap: {np.round(maximum_gap, 4)} for k={optimal_k}", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", line_width=2)
        fig.show("png")
