from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

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