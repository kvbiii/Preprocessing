from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class Prediction_plots():
    def __init__(self):
        pass
    def compare_predictions_with_real_values(self, y_true, y_pred, metric):
        metrics = {"mse": torch.nn.MSELoss(),
                        "rmse": RMSELoss(),
                        "mae": torch.nn.L1Loss()}
        if metric not in metrics:
            raise ValueError('Unsupported metric: {}'.format(metric))
        eval_metric = metrics[metric]
        y_true = y_true.squeeze(1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(len(y_true))], y=y_true.flatten().tolist(), mode='lines', line=dict(color="orange"), name="Real values"))
        fig.add_trace(go.Scatter(x=[i for i in range(len(y_true))], y=y_pred.flatten().tolist(), mode='lines', line=dict(color="blue"), name="Predictions"))
        fig.update_layout(template="simple_white", width=600, height=600, title="Predictions and Real values", xaxis_title="", yaxis_title="Values", font=dict(family="Times New Roman",size=16,color="Black"), legend_title_text='{}: {}'.format(metric.upper(), eval_metric(y_true, y_pred).item()))
        fig.show("png")

    def plot_losses(self, train_loss, valid_loss):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(len(train_loss))], y=train_loss, mode='lines', line=dict(color="orange"), name="Train loss"))
        fig.add_trace(go.Scatter(x=[i for i in range(len(valid_loss))], y=valid_loss, mode='lines', line=dict(color="blue"), name="Valid loss"))
        fig.update_layout(template="simple_white", width=600, height=600, title="Loss plot", xaxis_title="Epochs", yaxis_title="Loss", font=dict(family="Times New Roman",size=16,color="Black"), legend=dict(yanchor="top",y=0.85,xanchor="left",x=0.65))
        fig.show("png")
    
    def plot_feature_importances(self, feature_importances, column_names=None, nlargest=None):
        if(column_names == None):
            column_names = np.array([i for i in range(len(feature_importances))])
        if not isinstance(column_names, np.ndarray):
            try:
                column_names = np.array(column_names)
            except:
                raise TypeError('Wrong type of column_names. It should be numpy array, or list..')
        if(nlargest != None):
            ranking = (np.argsort(np.argsort(-np.array(feature_importances))))
            support = np.where(ranking < nlargest, True, False)
            feature_importances = feature_importances[support]
            column_names = column_names[support]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=column_names, y=feature_importances, marker_color='rgb(26, 118, 255)'))
        fig.update_layout(template="simple_white", width=max(30*len(column_names), 600), height=max(30*len(column_names), 600), title_text="<b>Feature importance<b>", title_x=0.5, yaxis_title="Feature importance", xaxis=dict(title='Features', showticklabels=True, type="category"), font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def conf_matrix(self, y_true, y_pred, normalize=False):
        if(normalize==True):
            CM = confusion_matrix(y_true, y_pred, normalize='true')
        else:
            CM = confusion_matrix(y_true, y_pred, normalize=None)
        TN = np.round(CM[0][0], 3)
        FN = np.round(CM[1][0], 3)
        TP = np.round(CM[1][1], 3)
        FP = np.round(CM[0][1], 3)
        z = [[TN, FP],
            [FN, TP]]
        z_text = [[str(y) for y in x] for x in z]
        x = ['0', '1']
        y =  ['0', '1']
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='blues')
        fig.update_layout(template="simple_white", width=600, height=600, showlegend=False, font=dict(family="Times New Roman",size=16,color="Black"), title_text="<b>Normalized confusion matrix<br>Balanced accuracy score: {}<b>".format(np.round(balanced_accuracy_score(y_true, y_pred), 4)), title_x=0.5, title_y=0.97)
        fig.add_annotation(dict(font=dict(family="Times New Roman",size=20,color="Black"),x=-0.15,y=0.5,showarrow=False,text="Actual",textangle=-90,xref="paper",yref="paper"))
        fig.add_annotation(dict(font=dict(family="Times New Roman",size=20,color="Black"),x=0.5,y=1.1,showarrow=False,text="Predictions",xref="paper",yref="paper"))
        fig['data'][0]['showscale'] = True
        fig.show("png")