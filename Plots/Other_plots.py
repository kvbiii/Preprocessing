from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
from sklearn.neighbors import NearestNeighbors

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
    
    def scatter_plot(self, X, y):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode="markers", marker=dict(color=y, colorscale="Viridis", showscale=False)))
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Scatter Plot<b>", title_x=0.5, xaxis_title="X_1", yaxis_title="X_2", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")

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
    
    def scree_plot(self, fitted_fa):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(fitted_fa.eigenvalues_)+1), y=fitted_fa.eigenvalues_, mode="lines+markers", marker=dict(line=dict(color='black', width=1)),))
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", line_width=3)
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Scree Plot<b>", title_x=0.5, xaxis_title="Component number", yaxis_title="Eigenvalues", font=dict(family="Times New Roman",size=16,color="Black"))
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

    def cross_validation_components(self, X, y, list_of_categorical, reduction_algorithm, predictive_algorithm, cv, random_state):
        list_of_continous = [feature for feature in X.columns.tolist() if feature not in list_of_categorical]
        fig = go.Figure()
        for n_components in range(1, len(list_of_continous)+1):
            reduction_algorithm.set_params(n_components=n_components, random_state=random_state)
            train_scores, valid_scores = [], []
            for iter, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                X_train, X_valid = X.iloc[train_idx, :], X.iloc[valid_idx, :]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                continous_data_train = X_train.drop(list_of_categorical, axis=1)
                continous_data_valid = X_valid.drop(list_of_categorical, axis=1)
                X_train_continous_transformed = reduction_algorithm.fit_transform(continous_data_train)
                X_valid_continous_transformed = reduction_algorithm.transform(continous_data_valid)
                X_train_transformed = np.concatenate([X_train_continous_transformed, X_train[list_of_categorical]], axis=1)
                X_valid_transformed = np.concatenate([X_valid_continous_transformed, X_valid[list_of_categorical]], axis=1)
                predictive_algorithm.fit(X_train_transformed, y_train)
                y_train_prob = predictive_algorithm.predict_proba(X_train_transformed)[:, 1]
                y_valid_prob = predictive_algorithm.predict_proba(X_valid_transformed)[:, 1]
                train_scores.append(roc_auc_score(y_train, y_train_prob))
                valid_scores.append(roc_auc_score(y_valid, y_valid_prob))
            fig.add_trace(go.Box(y=valid_scores, name=str(n_components), marker=dict(line=dict(color='black', width=1)), showlegend=False))
        fig.update_layout(template="simple_white", width=max(30*len(list_of_continous), 600), height=max(30*len(list_of_continous), 600), title=f"<b>Box Plot of valid scores<b>", title_x=0.5, yaxis_title="Valid Scores", xaxis_title="Number of components", xaxis=dict(showticklabels=True), font=dict(family="Times New Roman",size=16,color="Black"))
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
    
    def elbow_original_plot(self, data, algorithm_instance, max_clusters=15):
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[k for k in range(1, max_clusters+1)], y=[i for i in sse], mode='lines+markers', name="Original curve"))
        fig.add_trace(go.Scatter(x=[1, max_clusters], y=[np.max(sse), 0], mode='lines', line=dict(color="green", dash='dash'), name="Diagonal line"))
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0.75, max_clusters+0.5], constrain='domain', linecolor='black')
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Number of clusters (k)", yaxis_title="Distortion Score", legend=dict(x=0.75, y=0.9), showlegend=True, title=f"<b>Original elbow<b>", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def elbow_normalized_plot(self, data, algorithm_instance, max_clusters=15):
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
        scaled_sse = (sse-np.min(sse))/(np.max(sse)-np.min(sse))
        cluster_range = [k for k in range(1, max_clusters+1)]
        scaled_cluster_range = (cluster_range-np.min(cluster_range))/(np.max(cluster_range)-np.min(cluster_range))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scaled_cluster_range, y=scaled_sse, mode='lines+markers', name="Normalized curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', line=dict(color="green", dash='dash'), name="Diagonal line"))
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0, 1.05], constrain='domain', linecolor='black')
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Normalized number of clusters (k)", yaxis_title="Normalized distortion Score", legend=dict(x=0.75, y=0.9), showlegend=True, title=f"<b>Normalized elbow<b>", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def elbow_normalized_plot_with_perpendicular(self, data, algorithm_instance, max_clusters=15):
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
        scaled_sse = (sse-np.min(sse))/(np.max(sse)-np.min(sse))
        cluster_range = [k for k in range(1, max_clusters+1)]
        scaled_cluster_range = (cluster_range-np.min(cluster_range))/(np.max(cluster_range)-np.min(cluster_range))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scaled_cluster_range, y=scaled_sse, mode='lines+markers', name="Normalized curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', line=dict(color="green", dash='dash'), name="Diagonal line"))
        i = 0
        a=1
        b=1
        c=-1
        show=True
        while(i < len(scaled_sse)):
            x_0 = (b*(b*scaled_cluster_range[i]-a*scaled_sse[i])-a*c)/(a**2+b**2)
            y_0 = (a*(-b*scaled_cluster_range[i]+a*scaled_sse[i])-b*c)/(a**2+b**2)
            if(i > 0):
                show=False
            fig.add_trace(go.Scatter(x=[scaled_cluster_range[i], x_0], y=[scaled_sse[i], y_0], mode='lines+markers', marker=dict(color="grey"), name="Perpendicular lines", showlegend=show))
            i = i + 1
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0, 1.05], constrain='domain', linecolor='black')
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Normalized number of clusters (k)", yaxis_title="Normalized distortion Score", legend=dict(x=0.75, y=0.9), showlegend=True, title=f"<b>Normalized elbow with perpendicular lines<b>", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def elbow_normalized_plot_with_perpendicular_rotated(self, data, algorithm_instance, max_clusters=15):
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
        scaled_sse = (sse-np.min(sse))/(np.max(sse)-np.min(sse))
        cluster_range = [k for k in range(1, max_clusters+1)]
        scaled_cluster_range = (cluster_range-np.min(cluster_range))/(np.max(cluster_range)-np.min(cluster_range))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scaled_cluster_range, y=scaled_sse, mode='lines+markers', name="Normalized curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', line=dict(color="green", dash='dash'), name="Diagonal line"))
        i = 0
        a=1
        b=1
        c=-1
        show=True
        while(i < len(scaled_sse)):
            x_0 = (b*(b*scaled_cluster_range[i]-a*scaled_sse[i])-a*c)/(a**2+b**2)
            y_0 = (a*(-b*scaled_cluster_range[i]+a*scaled_sse[i])-b*c)/(a**2+b**2)
            if(i > 0):
                show=False
            fig.add_trace(go.Scatter(x=[scaled_cluster_range[i], x_0], y=[scaled_sse[i], y_0], mode='lines+markers', marker=dict(color="grey"), name="Perpendicular lines", showlegend=show))
            fig.add_trace(go.Scatter(x=[scaled_cluster_range[i], scaled_cluster_range[i]], y=[scaled_sse[i], 1-scaled_cluster_range[i]], mode='lines+markers', marker=dict(color="gold"), name="Perpendicular lines (rotated by 45 degrees)", showlegend=show))
            i = i + 1
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0, 1.05], constrain='domain', linecolor='black')
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Normalized number of clusters (k)", yaxis_title="Normalized distortion Score", legend=dict(x=0.3, y=0.9), showlegend=True, title=f"<b>Normalized elbow and perpendicular lines<br>(rotated by 45 degrees)<b>", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def difference_curve(self, data, algorithm_instance, max_clusters=15):
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
        scaled_sse = (sse-np.min(sse))/(np.max(sse)-np.min(sse))
        cluster_range = [k for k in range(1, max_clusters+1)]
        scaled_cluster_range = (cluster_range-np.min(cluster_range))/(np.max(cluster_range)-np.min(cluster_range))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scaled_cluster_range, y=scaled_sse, mode='lines+markers', name="Normalized curve"))
        fig.add_trace(go.Scatter(x=scaled_cluster_range, y=(1-scaled_cluster_range)-scaled_sse, mode='lines+markers', line=dict(color="orange"), name="Difference curve"))
        maximum_indice = np.argmax((1-scaled_cluster_range)-scaled_sse)
        optimal_sse = sse[maximum_indice]
        optimal_k = cluster_range[maximum_indice]
        fig.add_vline(x=scaled_cluster_range[maximum_indice], line_dash="dash", line_color="red", line_width=2)
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0, 1.05], constrain='domain', linecolor='black')
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Normalized number of clusters (k)", yaxis_title="Normalized distortion Score", legend=dict(x=0.75, y=0.9), showlegend=True, title=f"<b>Difference curve<b><br>Optimal SSE: {np.round(optimal_sse, 4)} for k={optimal_k}", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
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
    
    def silhouette_plot_for_various_clusters(self, data, algorithm_instance, max_clusters=10):
        data = self.check_2d_data(data=data)
        all_number_of_samples = data.shape[0]
        for k in range(2, max_clusters+1):
            algorithm_instance.set_params(n_clusters=k)
            algorithm_instance.fit(data)
            silhouette_coefficients = self.calculate_silhouette_coefficient(data=data, algorithm_instance=algorithm_instance)
            average_silhouette_coefficient = np.mean(silhouette_coefficients)
            y_lower = int(all_number_of_samples/5)
            fig = go.Figure()
            for cluster in np.unique(algorithm_instance.labels_):
                indices_of_current_cluster = np.where(algorithm_instance.labels_ == cluster)[0]
                silhouette_coefficients_of_current_cluster = np.sort(silhouette_coefficients[indices_of_current_cluster])
                y_upper = y_lower + len(data[np.where(algorithm_instance.labels_ == cluster)])
                fig.add_trace(go.Scatter(x=silhouette_coefficients_of_current_cluster, y=np.arange(y_lower, y_upper), fill='tozerox', mode='lines'))
                fig.add_annotation(x=0.2, y=np.mean([y_lower, y_upper])-0.5, text=str(cluster), showarrow=False, arrowhead=1, font=dict(family="Times New Roman",size=25,color="Black"), yshift=0)
                y_lower = y_upper+int(all_number_of_samples*0.02)
            fig.add_vline(x=average_silhouette_coefficient, line_dash="dash", line_color="red", line_width=2)
            fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Silhouette coefficient", yaxis_title="Cluster label", yaxis_showticklabels=False, yaxis_visible=False, showlegend=False, title=f"<b>Silhouette plot for various clusters<b><br>Average Silhouette Coefficient: {np.round(average_silhouette_coefficient, 4)}", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
            fig.show("png")
    
    def calculate_silhouette_coefficient(self, data, algorithm_instance):
        algorithm_instance.fit(data)
        silhouette_coefficients = np.zeros(shape=(data.shape[0],))
        for cluster in np.unique(algorithm_instance.labels_):
            indices_of_current_cluster = np.where(algorithm_instance.labels_ == cluster)[0]
            X_cluster = data[indices_of_current_cluster]
            if(X_cluster.shape[0] == 1):
                silhouette_coefficients[indices_of_current_cluster] = [0]
                continue
            a_i = 1/(X_cluster.shape[0]-1)*np.sum(np.sqrt(np.sum((X_cluster[:,np.newaxis]-X_cluster)**2, axis=2)), axis=1)
            min_bi = np.array([np.inf for i in range(X_cluster.shape[0])])
            for other_cluster in np.unique(algorithm_instance.labels_):
                if(other_cluster != cluster):
                    X_other_cluster = data[np.where(algorithm_instance.labels_ == other_cluster)]
                    b_i = 1/(X_other_cluster.shape[0])*np.sum(np.sqrt(np.sum((X_cluster[:,np.newaxis]-X_other_cluster)**2, axis=2)), axis=1)
                    min_bi = np.min([min_bi, b_i], axis=0)
            silhouette_coefficients[indices_of_current_cluster] = list((min_bi-a_i)/np.max([a_i, min_bi], axis=0))
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
    
    def plot_cluster_2d_data(self, data, model):
        data = self.check_2d_data(data=data)
        X_1 = data[:,0]
        X_2 = data[:,1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_1, y=X_2, mode='markers', marker=dict(color=model.labels_, colorscale='Viridis', line=dict(color='black', width=1))))
        fig.add_trace(go.Scatter(x=model.cluster_centers_[:,0], y=model.cluster_centers_[:,1], mode='markers', marker=dict(color='red', symbol='x', size=12, line=dict(color='black', width=1))))
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="X_1", yaxis_title="X_2", showlegend=False, title="<b>Clustered data<b>", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def plot_cluster_2d_data_without_centers(self, data, model):
        data = self.check_2d_data(data=data)
        X_1 = data[:,0]
        X_2 = data[:,1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_1, y=X_2, mode='markers', marker=dict(color=model.labels_, colorscale='Viridis', line=dict(color='black', width=1))))
        fig.update_layout(template="simple_white", width=800, height=800, xaxis_title="X_1", yaxis_title="X_2", showlegend=False, title="<b>Clustered data<b>", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def k_distance_plot(self, data, min_points):
        data = self.check_2d_data(data=data)
        model = NearestNeighbors(n_neighbors=min_points, metric="euclidean")
        model.fit(data)
        distances, indices = model.kneighbors(data)
        distances = distances[:, 1:]
        mean_of_distances = np.mean(distances, axis=1)
        sorted_mean_of_distances = np.sort(mean_of_distances)[::-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(0, len(sorted_mean_of_distances))], y=sorted_mean_of_distances, mode='lines', name="Original curve"))
        fig.add_trace(go.Scatter(x=[1, len(sorted_mean_of_distances)], y=[np.max(sorted_mean_of_distances), np.min(sorted_mean_of_distances)], mode='lines', line=dict(color="green", dash='dash'), name="Diagonal line"))
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0.75, len(sorted_mean_of_distances)+0.5], constrain='domain', linecolor='black')
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Mean distance to neighbors for each observation (descending)", yaxis_title="Mean distance", legend=dict(x=0.75, y=0.9), showlegend=True, title=f"<b>K-distance plot<b>", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def k_distance_normalized_plot(self, data, min_points):
        data = self.check_2d_data(data=data)
        model = NearestNeighbors(n_neighbors=min_points, metric="euclidean")
        model.fit(data)
        distances, indices = model.kneighbors(data)
        distances = distances[:, 1:]
        mean_of_distances = np.mean(distances, axis=1)
        sorted_mean_of_distances = np.sort(mean_of_distances)[::-1]
        scaled_sorted_mean_of_distances = (sorted_mean_of_distances-np.min(sorted_mean_of_distances))/(np.max(sorted_mean_of_distances)-np.min(sorted_mean_of_distances))
        data_range = [k for k in range(0, len(scaled_sorted_mean_of_distances))]
        scaled_data_range = (data_range-np.min(data_range))/(np.max(data_range)-np.min(data_range))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scaled_data_range, y=scaled_sorted_mean_of_distances, mode='lines', name="Normalized curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', line=dict(color="green", dash='dash'), name="Diagonal line"))
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0, 1.05], constrain='domain', linecolor='black')
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Normalized sorted mean distances (descending)", yaxis_title="Normalized mean distances", legend=dict(x=0.75, y=0.9), showlegend=True, title=f"<b>Normalized K-distance plot<b>", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def difference_curve_epsilon(self, data, min_points):
        data = self.check_2d_data(data=data)
        model = NearestNeighbors(n_neighbors=min_points, metric="euclidean")
        model.fit(data)
        distances, indices = model.kneighbors(data)
        distances = distances[:, 1:]
        mean_of_distances = np.mean(distances, axis=1)
        sorted_mean_of_distances = np.sort(mean_of_distances)[::-1]
        scaled_sorted_mean_of_distances = (sorted_mean_of_distances-np.min(sorted_mean_of_distances))/(np.max(sorted_mean_of_distances)-np.min(sorted_mean_of_distances))
        data_range = [k for k in range(0, len(scaled_sorted_mean_of_distances))]
        scaled_data_range = (data_range-np.min(data_range))/(np.max(data_range)-np.min(data_range))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scaled_data_range, y=scaled_sorted_mean_of_distances, mode='lines', name="Normalized curve"))
        fig.add_trace(go.Scatter(x=scaled_data_range, y=(1-scaled_data_range)-scaled_sorted_mean_of_distances, mode='lines', line=dict(color="orange"), name="Difference curve"))
        maximum_indice = np.argmax((1-scaled_data_range)-scaled_sorted_mean_of_distances)
        optimal_epsilon = sorted_mean_of_distances[maximum_indice]
        fig.add_vline(x=scaled_data_range[maximum_indice], line_dash="dash", line_color="red", line_width=2)
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0, 1.05], constrain='domain', linecolor='black')
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Normalized sorted mean distances (descending)", yaxis_title="Normalized mean distances", legend=dict(x=0.75, y=0.9), showlegend=True, title=f"<b>Difference curve<b><br>Optimal epsilon: {np.round(optimal_epsilon, 4)}", title_x=0.5, font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
