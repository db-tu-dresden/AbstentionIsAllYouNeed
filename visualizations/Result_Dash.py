from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import re
import numpy as np
import pandas as pd


with open('evaluation_full.pkl', 'rb') as f:
    df = pickle.load(f)

with open('evaluation_time_full.pkl', 'rb') as f:
    df_time = pickle.load(f)

with open('evaluation_ranking_full.pkl', 'rb') as f:
    df_rank = pickle.load(f)
    
with open('evaluation_roc.pkl', 'rb') as f:
    df_roc = pickle.load(f)


rejectors = df[df['Method'].str.contains(
    'Rejector', case=False, na=False)]['Method'].unique()
embeddings = df[df['Method'].str.contains(
    'Embedding', case=False, na=False)]['Method'].unique()
classification = df[df['Method'].str.contains('Dropout', case=False, na=False) |
                    df['Method'].str.contains('Softmax', case=False, na=False)]['Method'].unique()
outlier = df[df['Method'].str.contains(
    'Outlier', case=False, na=False)]['Method'].unique()
exclude_keywords = ['Rejector', 'Embedding', 'Dropout', 'Softmax', 'Outlier']
other_methods = df[~df['Method'].str.contains(
    '|'.join(exclude_keywords), case=False, na=False)]['Method'].unique()


def find_intersection(a, b, c, d):
    if b == d:
        return float('-inf')
    else:
        x = (c - a) / (b - d)
        if x <= 0:
            return float('-inf')
        else:
            return x


app = Dash()

app.layout = [
    html.H1(children='Results', style={'textAlign': 'center'}),
    html.P("Select Dataset"),
    dcc.Dropdown(id="dataset", options=df[~df['Dataset'].str.contains(
        "Interpolated")]['Dataset'].unique(), value="AG_News_PHI", clearable=False),
    # dcc.Checklist(df.Method.unique(), ["TinyBert Rejector","Most Frequent Class","Tf-Idf Embedding","Dropout Softmax Variance","Softmax"], id='method-selection', inline=True),# [method for method in df.Method.unique() if method not in ["Base Embeddings", "Base Outlier"]]

    html.Div(
        [
            html.P("Classification Only Methods", style={
                   "margin-right": "10px", "font-weight": "bold"}),
            dcc.Checklist(classification, [
                          "Softmax", "Dropout Softmax MV"], id='method-selection', inline=True),
        ],
        style={"display": "flex", "align-items": "center",
               "margin-bottom": "0px"},
    ),
    html.Div(
        [
            html.P("Embedding-Matchvector Methods",
                   style={"margin-right": "10px", "font-weight": "bold"}),
            dcc.Checklist(embeddings, ["Base Embedding", "Tf-Idf Embedding"],
                          id='embedding-selection', inline=True),
        ],
        style={"display": "flex", "align-items": "center",
               "margin-bottom": "0px"},
    ),
    html.Div(
        [
            html.P("Isolation Forest Methods", style={
                   "margin-right": "10px", "font-weight": "bold"}),
            dcc.Checklist(outlier, [], id='outlier-selection', inline=True),
        ],
        style={"display": "flex", "align-items": "center",
               "margin-bottom": "0px"},
    ),
    html.Div(
        [
            html.P("Finetuned Rejector", style={
                   "margin-right": "10px", "font-weight": "bold"}),
            dcc.Checklist(rejectors, ["Distilbert Rejector"],
                          id='rejector-selection', inline=True),
        ],
        style={"display": "flex", "align-items": "center",
               "margin-bottom": "0px"},
    ),
    html.Div(
        [
            html.P("Other", style={
                   "margin-right": "10px", "font-weight": "bold"}),
            dcc.Checklist(other_methods, other_methods,
                          id='other-selection', inline=True),

        ],
        style={"display": "flex", "align-items": "center",
               "margin-bottom": "0px"},
    ),
    html.Div(
        [
            html.P("Classifier Type", style={
                   "margin-right": "10px", "font-weight": "bold"}),
            dcc.Checklist(["SVM", "Random Forest", "Logistic Regression", "Simple NN", "generic"], [
                          "Simple NN", "generic"], id='slider_classifier', inline=True),

        ],
        style={"display": "flex", "align-items": "center",
               "margin-bottom": "0px"},
    ),


    html.P('Slider:', style={'font-weight': 'bold'}),
    dcc.RadioItems(['Interpolated', 'Raw'], 'Interpolated',
                   id='Interpolated', inline=True),

    html.Div(
        style={'display': 'flex', 'margin-top': '20px'},
        children=[
            dcc.Graph(id='graph-content',
                      style={'width': '1000px', 'height': '400px'}),
            dcc.Graph(id='roc_fig',
                style={'width': '800px', 'height': '400px'}),
            html.Div(
                id='explanation-window',
                style={
                    'border': '1px solid black',
                    'padding': '10px',
                    'margin-left': '20px',
                    'width': '200px',
                    'height': '150px',
                    'overflow-y': 'auto'
                }
            )
        ]
    ),

    html.Div(

        [
            html.P("UARC Score Range", style={
                   "margin-right": "10px", "font-weight": "bold", 'width': '10%'}),
            html.Div(
                dcc.Slider(id='score_slider', min=0.4, max=1, step=None,
                           marks={0.5: '1.0 to 0.5 Coverage', 0.9: '1.0 to 0.9 Coverage', 0.95: '1.0 to 0.95 Coverage'}, value=0.9),
                # Ensure this div takes full width
                style={"width": "100%", "margin-right": "300px"},
            ),

        ],
        style={'display': 'flex', 'align-items': 'center', 'width': '100%'},
    ),

    html.Div(

        [
            html.P("Dataset Size Range", style={
                   "margin-right": "10px", "font-weight": "bold", 'width': '10%'}),
            html.Div(
                dcc.RangeSlider(id='range_slider', min=100, max=10000, step=None,
                                marks={500: '500', 1000: '1k', 2000: '2k', 4000: '4k', 6000: '6k', 8000: '8k', 10000: '10k'}, value=[10000, 10000]),
                # Ensure this div takes full width
                style={"width": "100%", "margin-right": "100px"},
            ),

        ],
        style={'display': 'flex', 'align-items': 'center', 'width': '100%'},
    ),

    dcc.Graph(id='graph-content_2',
              style={'width': '1600px', 'height': '600'}),
    dcc.Checklist(['Show Grid', 'Cropped X-Axis'],
                  ['Cropped X-Axis'], id='settings', inline=True),

    html.Div(
        [
            html.P("Method Time Complexity", style={
                   "margin-right": "10px", "font-weight": "bold", 'width': '10%'}),
            html.Div(
                [
                    dcc.Dropdown(id="method1", options=[{'label': i, 'value': i} for i in df.Method_Classifier.unique(
                    )], value="Softmax generic", clearable=False),
                    dcc.Dropdown(id="method2", options=[{'label': i, 'value': i} for i in df.Method_Classifier.unique(
                    )], value="Distilbert Rejector generic", clearable=False),
                    dcc.Dropdown(id="method3", options=[{'label': i, 'value': i} for i in df.Method_Classifier.unique(
                    )], value="Dropout Softmax MV generic", clearable=False),
                ],
                # Ensure this div takes full width
                style={"width": "100%", "margin-right": "100px"},
            ),
        ],
        style={'display': 'flex', 'align-items': 'center'},
    ),

    html.Div(
        style={'display': 'flex', 'margin-top': '20px'},
        children=[
            dcc.Graph(id='Comparison-graph',
                      style={'width': '1000px', 'height': '400px'}),
            html.Div(
                id='break_even_window',
                style={
                    'padding': '10px',
                    'margin-left': '20px',
                    'width': '400px',
                    'height': '350px',
                    'overflow-y': 'auto'
                }
            )
        ]
    ),#
    


    html.H1("Rank Table"),
    dash_table.DataTable(
        id='rank-table',
        columns=[{"name": col, "id": col}
                 for col in df_rank.columns],  # Columns for the table
        data=df_rank.to_dict('records'),  # Data from the DataFrame
        style_table={'overflowX': 'auto'},  # Style for scrollable table
        style_cell={'textAlign': 'left', 'padding': '5px'},  # Cell styling
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        }
    )
    # html.Div(id='explanation-window', style={'border': '1px solid black', 'padding': '10px', 'margin-top': '10px'})#





]


@app.callback(
    Output('graph-content', 'figure'),
    Output('graph-content_2', 'figure'),
    Output('explanation-window', 'children'),
    Output("break_even_window", 'children'),
    Output('Comparison-graph', 'figure'),
    Output('roc_fig', 'figure'),
    Output('rank-table', 'data'),
    Input('method-selection', 'value'),
    Input('embedding-selection', 'value'),
    Input('outlier-selection', 'value'),
    Input('rejector-selection', 'value'),
    Input('other-selection', 'value'),
    Input('settings', 'value'),
    Input('range_slider', 'value'),
    Input('dataset', 'value'),
    Input('slider_classifier', 'value'),
    Input('score_slider', 'value'),
    Input('Interpolated', 'value'),
    Input('method1', 'value'),
    Input('method2', 'value'),
    Input('method3', 'value'),
)
# ich schätze die sind nach reihenfolge der "values" bei callback
def update_graph(methods, methodsB, methodsC, methodsD, methodsE, settings, range_slider, dataset, classifier, score_slider, format, method1, method2, method3):

    methods = methods + methodsB + methodsC + methodsD + methodsE
    # methods = np.concatenate((methods,methodsB,methodsC,methodsD,methodsE), ignore_index=True)

    if dataset == "Sentiment":
        explanation = "Multiclass (pos,neutral,neg) Sentiment Analysis Dataset"
    elif dataset == "Question Answering":
        explanation = "Context-Question-Answer (SQUAD) Dataset"
    elif dataset == "AG_News":
        explanation = "AG News Classification Dataset"
    elif dataset == "Spam":
        explanation = "Email Spam Classification Dataset mit Email TITEL als Input"
    elif dataset == "Spam_text":
        explanation = "Email Spam Classification Dataset mit Email INHALT als Input"
    elif dataset == "Regression":
        explanation = "Movie Review Sentiment als Regression - Model predicted Sternenanzahl als float value"
    elif dataset == "Transformation":
        explanation = "Spell Checker - Model korrigert Rechtschreibung"
    elif dataset == "Time Series Regression":
        explanation = "Periodic Time Series data - Input: 5 Values, Predict the next - Correct if distance to label is close enough"
    elif dataset == "Merged":
        explanation = "Alle Interpolated Datasets Merged, ohne RunTime Merge"
        format = "Raw"  # Merged ist eigentlich Interpolated Merge, aber für Anzeige irrelevant
    elif "GLUE" in dataset:
        max_size = df[df['Dataset']==dataset]['Data_size'].iloc[0]
        print("Glue Datasize:",df[df['Dataset']==dataset]['Data_size'].iloc[0])
        explanation = "GLUE DATASET, CUSTOM DATA SIZE: " + str(max_size) 
        range_slider[0] = 0 #Da nur eine Datasize, halt genau die dann
        range_slider[1] = max_size
    else:
        explanation = "Erklärung kommt noch"
        #

    if format == "Interpolated":
        dataset = dataset + " Interpolated"

    dff = df[df['Dataset'] == dataset]
    dff = dff[dff.Method.isin(methods)]
    dff = dff[dff.Classifier.isin(classifier)]

    # Method Classifier Stuff
    method_classifier_array = dff.Method_Classifier.unique()
    # Distilbert,Softmax,Dropout prob am besten zum vergleichen

    dff = dff[(dff['Data_size'] >= range_slider[0]) &
              (df['Data_size'] <= range_slider[1])]

    if 'Cropped X-Axis' in settings:
        dff = dff[dff['Coverage'] >= score_slider] #0.5 original
        # fig['layout']['xaxis'] = {'range': (1, 0.5)}

    fig = px.line(dff, x="Coverage", y="Accuracy", color='Method_Classifier',
                  line_group='Data_size', line_dash='Data_size')
    fig.update_xaxes(autorange='reversed')  # Invert y-axis if desired
    fig.update_layout(plot_bgcolor='white')

    # Update grid lines based on checkbox selection
    show_grid = 'Show Grid' in settings
    fig.update_xaxes(showgrid=show_grid, gridcolor='grey')  # X-axis grid color
    fig.update_yaxes(showgrid=show_grid, gridcolor='grey')  # Y-axis grid color

    # Run Time Graphen
    # ----------------------------------------------------------------------------
    if format == "Raw":
        # Scores und Runtime Values innerhalb Interpolated Data gespeichert, nur zum Zugreifen
        dff_time = df_time[df_time['Dataset'] == dataset + " Interpolated"]
    else:
        dff_time = df_time[df_time['Dataset'] == dataset]

    dff_time = dff_time[(dff_time['Data_size'] >= range_slider[0]) & (
        dff_time['Data_size'] <= range_slider[1])]

    dff_time = dff_time[dff_time.Method.isin(methods)]
    dff_time = dff_time[dff_time.Classifier.isin(classifier)]

    # Create subplots: 1 row, 2 columns
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=(
        "Inference Time vs Score", "Initial Time vs Score"))

    # Create scatter plots

    if score_slider == 0.95:
        score_slider_val = "Score_95"
    elif score_slider == 0.90:
        score_slider_val = "Score_90"
    else:
        score_slider_val = "Score_50"

    inference_fig = px.scatter(dff_time, x=score_slider_val,
                               y="Inference Time", color='Method', symbol='Data_size')
    initial_fig = px.scatter(dff_time, x=score_slider_val,
                             y="Initial Time", color='Method', symbol='Data_size')

    # Add traces to subplots
    for trace in inference_fig.data:
        fig2.add_trace(trace, row=1, col=1)

    for trace in initial_fig.data:
        fig2.add_trace(trace, row=1, col=2)

    fig2.update_layout(
        title_text="Score X Inference and Initial Time X Data sizes(Range Slider auswählen)",)
    fig2.update_xaxes(title_text="Score", row=1, col=1)
    fig2.update_yaxes(title_text="Inference Time in seconds", row=1, col=1)
    fig2.update_xaxes(title_text="Score", row=1, col=2)
    fig2.update_yaxes(title_text="Initial Time in seconds", row=1, col=2)





    # Method Time Comparison
    # ------------------------------------------------------------------------------
    #print(range_slider[1], dataset)
    dcomp_base_time = df_time[(df_time['Dataset'] == dataset) & (
        df_time['Data_size'] == range_slider[1])]



    possible_selection = dcomp_base_time.Method_Classifier.unique()
    
    if method1 in possible_selection and method2 in possible_selection and method3 in possible_selection:
        coeff = []
        functions = {}
        for method in [method1, method2, method3]:
            a = dcomp_base_time[dcomp_base_time["Method_Classifier"]
                                == method]["Initial Time"].values[0]
            b = dcomp_base_time[dcomp_base_time["Method_Classifier"]
                                == method]["Inference Time"].values[0]
            b = b/2000  # Inference 2k Data
            functions[method] = lambda x, a=a, b=b: a + b * x
            #print(method)
            #print(f"{a}+{b}x")
            coeff.append([a, b])

        intersection1_2 = find_intersection(
            coeff[0][0], coeff[0][1], coeff[1][0], coeff[1][1])
        intersection2_3 = find_intersection(
            coeff[2][0], coeff[2][1], coeff[1][0], coeff[1][1])

        scale_graph_x = max(intersection1_2, intersection2_3)*1.1
        # Generate x values
        x = np.linspace(0, scale_graph_x, 1000)

        # Prepare a DataFrame for Seaborn
        data = {"Datapoints": [], "Time in seconds": [], "Method": []}
        for name, func in functions.items():
            y = func(x)
            data["Datapoints"].extend(x)
            data["Time in seconds"].extend(y)
            data["Method"].extend([name] * len(x))

        # Create the DataFrame
        df_comp_graph = pd.DataFrame(data)

        # Initialize figure and axis
        comparison_graph_fig = px.line(
            df_comp_graph, y="Time in seconds", x="Datapoints", color="Method")
        comparison_graph_fig.update_layout(plot_bgcolor='white')

        break_even_window = (
            #f"{method1}: {coeff[0][0]:.2f}s + {coeff[0][1]:.4f}x/s\n"
            #f"{method2}: {coeff[1][0]:.2f}s + {coeff[1][1]:.4f}x/s\n"
            #f"{method3}: {coeff[2][0]:.2f}s + {coeff[2][1]:.4f}x/s\n"
            f"Break Even: \n"
            f"{method1},{method2}:{intersection1_2:.0f}\n Datapoints and "
            f"{method2},{method3}:{intersection2_3:.0f} Datapoints"
        )
    else:
        break_even_window = "Ausgewählte Methode(n) für Modell nicht verfügbar"
        comparison_graph_fig = px.line(y=[0], x=[0])
        comparison_graph_fig.update_layout(plot_bgcolor='white')
    # -------------------------------------------
    # method_classifier_array
    filtered_rank = df_rank[df_rank['Dataset'] == dataset]
    # filtered_rank = df_rank[df_rank.Method.isin(methods)]
    # filtered_rank = filtered_rank[filtered_rank.Classifier.isin(classifier)]
    #
    #print(df.Dataset.unique())
    
    
    #df_roc
    #----------------------
    print(dataset,range_slider[1])#dataset,datasize(bigger one)
    selected_roc = df_roc[dataset.replace(" Interpolated", "")][range_slider[1]]
    graph_roc_dict = {}
    for method in method_classifier_array:# selected methods
        if method in selected_roc.keys(): #available methods
            graph_roc_dict[method] = selected_roc[method]
            print(method,selected_roc[method]["roc_auc"])
            
    #roc pandas for viz
    data_roc = []
    for method, metrics in graph_roc_dict.items():
        for fpr, tpr in zip(metrics["FPR"], metrics["TPR"]):
            data_roc.append({"Method": method, "FPR": fpr, "TPR": tpr, "roc_auc": metrics["roc_auc"]})
    df_roc_viz = pd.DataFrame(data_roc)
    
    #print(df_roc_viz)
    roc_fig = px.line(df_roc_viz, x="FPR", y="TPR", color='Method')
    roc_fig.update_layout(showlegend=False)

    
    return fig, fig2, explanation, break_even_window, comparison_graph_fig,roc_fig, filtered_rank.to_dict('records')


if __name__ == '__main__':
    app.run(debug=True)
    # app.run_server(host='0.0.0.0', port=8080, debug=True)
