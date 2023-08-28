from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class Preprocessing_plots():
    def __init__(self):
        pass

    def check_data(self, data):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series) and not isinstance(data, np.ndarray) and not torch.is_tensor(data):
            raise TypeError('Wrong type of data. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        data = np.array(data)
        if(data.ndim == 2):
            data = data.squeeze()
        return data

    def hist_plot_with_kde(self, data, name="", bin_size=1):
        data = self.check_data(data=data)
        hist_data = [data]
        group_labels =[name]
        fig = ff.create_distplot(hist_data, group_labels=group_labels, show_rug=False, curve_type='kde', bin_size=bin_size)
        fig.update_layout(template="simple_white", width=600, height=600, showlegend=False, title=f"<b>Histogram {name.title()}<b>", title_x=0.5, xaxis_title="Data", yaxis_title="Density", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def boxplot(self, data, with_annotation=False, name=""):
        data = self.check_data(data=data)
        fig = go.Figure()
        fig.add_trace(go.Box(y=data, showlegend=False, name=name))
        if(with_annotation==True):
            for x in zip(["Min","Q1","Med","Q3","Max"], np.quantile(data, [0, 0.25, 0.5, 0.75, 1])):
                fig.add_annotation(x=0.4, y=x[1], text=x[0] + ": " + str(np.round(x[1], 3)), showarrow=False)
        fig.update_layout(template="simple_white", width=600, xaxis_title="", height=600, showlegend=False, title=f"<b>Box Plot<b>", title_x=0.5, yaxis_title="Data values", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def pie_chart(self, data, name=""):
        data = self.check_data(data=data)
        fig = go.Figure()
        labels, frequency = np.unique(data, return_counts=True)
        fig.add_trace(go.Pie(values=frequency, labels=labels, showlegend=True, textinfo='value+percent', hole=0.3, marker=dict(line=dict(color='#000000', width=2))))
        fig.update_layout(template="simple_white", width=600, xaxis_title="", height=600, showlegend=True, title=f"<b>Pie chart {name.title()}<b>", title_x=0.5, yaxis_title="Data values", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def barplot(self, data, name=""):
        data = self.check_data(data=data)
        fig = go.Figure()
        labels, frequency = np.unique(data, return_counts=True)
        fig.add_trace(go.Bar(x=labels, y=frequency, marker=dict(line=dict(color='black', width=1))))
        fig.update_layout(template="simple_white", width=max(30*len(labels), 600), height=max(30*len(labels), 600), title=f"<b>Bar chart {name.title()}<b>", title_x=0.5, yaxis_title="Frequency", xaxis=dict(title=f'{name.title()} Categories', showticklabels=True, type="category"), font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def correlation_plot(self, df, features_names):
        df = np.array(df)
        df = pd.DataFrame(df, columns=features_names)
        corr = np.round(df[df.columns.tolist()].corr(), 3)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        df_mask = corr.mask(mask)
        fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), x=df_mask.columns.tolist(), y=df_mask.columns.tolist(), colorscale=px.colors.diverging.RdBu,hoverinfo="none", showscale=True, ygap=1, xgap=1)
        fig.update_xaxes(side="bottom")
        fig.update_layout(width=1200, height=800,xaxis_showgrid=False,yaxis_showgrid=False,xaxis_zeroline=False,yaxis_zeroline=False,yaxis_autorange='reversed',template='plotly_white',font=dict(family="Times New Roman",size=12,color="Black"))
        # NaN values are not handled automatically and are displayed in the figure
        # So we need to get rid of the text manually
        for i in range(len(fig.layout.annotations)):
            if fig.layout.annotations[i].text == 'nan':
                fig.layout.annotations[i].text = ""
        fig.show("png")