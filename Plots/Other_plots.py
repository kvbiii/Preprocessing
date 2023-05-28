from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
class Other_Plots():
    def __init__(self) -> None:
        pass
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