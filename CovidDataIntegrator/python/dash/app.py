from dash import Dash, html, dcc, callback, Output, Input, State
import dash
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dash_table, no_update
from helpers.files_processing import calculate_embeddings, calculate_sim_matrix_dtype, \
    matching_hungarian
from helpers import tabs
from helpers import uploader
from celery import Celery
import dash
from celery.schedules import crontab
from dash.long_callback import CeleryLongCallbackManager
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from collections import OrderedDict

celery_app = Celery(__name__, broker='redis://redis:6379/0', backend='redis://redis:6379/1')
long_callback_manager = CeleryLongCallbackManager(celery_app)
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
external_stylesheets = [
    "http://fonts.googleapis.com/css?family=Lato&subset=latin,latin-ext",
    dbc.icons.FONT_AWESOME,
    dbc.themes.SKETCHY,
    dbc_css
]

app = Dash(__name__, external_stylesheets=external_stylesheets, use_pages=True, suppress_callback_exceptions=True)
app.title = 'Long Covid Data Integrator'
modal_loading_matching = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Matching..."),close_button=False),
                dbc.ModalBody(children=["Processing...   ",dbc.Spinner(color="dark")]),
                dbc.ModalFooter(html.Progress(id="progress_bar"))
            ],
            id="modal-loading",
            centered=True,
            is_open=False,
        )
modal_loading_processing = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Preprocessing Data..."),close_button=False),
                dbc.ModalBody(children=["Processing...   ",dbc.Spinner(color="dark")]),
            ],
            id="modal-loading-preprocessing",
            centered=True,
            is_open=False,
        )

modal_loading_exporting = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Exporting Data..."),close_button=False),
                dbc.ModalBody(children=["Integration Excecution...   ",dbc.Spinner(color="dark")]),
            ],
            id="modal-loading-exporting",
            centered=True,
            is_open=False,
        )

modal_alert_option_mismatch = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Error..."),close_button=False),
                dbc.ModalBody(id="modal-alert-body"),
            ],
            id="modal-alert-mismatch",
            centered=True,
            is_open=False,
        )
app.layout = dbc.Container([
    html.H1('Covid Data Integrator'),
    tabs.get_tabs(),
    modal_loading_matching,
    modal_loading_processing,
    modal_loading_exporting,
    modal_alert_option_mismatch,
    dcc.Store(id='first-dataset-store', data=[]),
    dcc.Store(id='second-dataset-store', data=[]),
    dcc.Store(id='first-dataset-info-store'),
    dcc.Store(id='second-dataset-info-store'),
    dcc.Store(id='first-dataset-info-store-matching'),
    dcc.Store(id='second-dataset-info-store-matching'),
    dcc.Store(id='first-dataset-categorical-mappings'),
    dcc.Store(id='second-dataset-categorical-mappings'),

])




@app.long_callback(
    Output('export-data-btn','style'),    
    Output('datatable-matching', 'columns'),
    Output('datatable-matching', 'data'),
    Output('datatable-matching', 'dropdown'),
    Output('datatable-matching-user','columns'),
    #Output('datatable-matching-user', 'data',allow_duplicate=True),
    Output('datatable-matching-user','dropdown'),
    Output('add-data-btn','style'),
    Input('start-matching-btn', 'n_clicks'),
    State('first-dataset-store', 'data'),
    State('first-dataset-info-store-matching', 'data'),
    State('second-dataset-store', 'data'),
    State('second-dataset-info-store-matching', 'data'),
    running=[
        (Output('modal-loading', 'is_open'), True, False),
        (
            Output("progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
    ],
    progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
    manager=long_callback_manager,
    prevent_initial_call=True,
)
def _start_matching(set_progress, nc, df1, info1, df2, info2):
    try:
        print("Starting to load the embeddings model...", flush=True)
        with tqdm(total=100, desc="Loading model") as pbar:
            model = SentenceTransformer('svalabs/german-gpl-adapted-covid')
            pbar.update(100)
    except:
        return no_update,"Error Downloading Embeddings Model, Please Try Again"
    
    print('starting matching', flush=True)
    if not nc or not df1 or not df2:
        return no_update,"No matching data available. Please provide input and try again."

    print('calculating Embeddings', flush=True)

    # Calculate embeddings and similarity matrix
    embeddings_1, embeddings_2 = calculate_embeddings(info1['questions'], info2['questions'], model)
    set_progress((str(20), str(100)))  # Progress 50% done
    print('calculating sim matrix', flush=True)
    
    sim_matrix = calculate_sim_matrix_dtype(embeddings_1, embeddings_2, info1, info2)
    set_progress((str(60), str(100)))  # Progress 80% done
    # Perform matching
    m1 = matching_hungarian(sim_matrix)
    highest_sim = m1.loc[m1['Similarity'] > 0.6].copy()
    set_progress((str(90), str(100)))  # Progress 90% done

    # Format similarity column to 2 decimal places
    highest_sim['Similarity'] = highest_sim['Similarity'].apply(lambda x: f"{x:.2f}")

    highest_sim['Dataframe_1'] = highest_sim['Dataframe_1'].str.split(',')
    highest_sim['Dataframe_2'] = highest_sim['Dataframe_2'].str.split(',')
    expanded_rows = []
    
    for _, row in highest_sim.iterrows():
        proposers = row['Dataframe_1']
        receivers = row['Dataframe_2']
    
        # Create cross product for each row
        for proposer in proposers:
            for receiver in receivers:
                expanded_rows.append({'Dataframe_1': proposer, 'Dataframe_2': receiver, 'Similarity': row['Similarity']})
    
    # Create a new dataframe with the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    expanded_df['Matching_Options'] = ['union'] * len(expanded_df)
    columns=[{
            "name": col,
            "id": col,
            "editable": False
        } if col != 'Matching_Options' else {
            "name": "Matching_Options",
            "id": "Matching_Options",
            "presentation": "dropdown",
            "editable": True,
        } for col in expanded_df.columns]
    data=expanded_df.to_dict('records')
    dropdown={
            'Matching_Options': {
                'options': [
                    {'label': 'Unify', 'value': 'union'},
                    {'label': 'Inverse And Unify', 'value': 'inverse'},
                    {'label': 'Concat', 'value': 'row bind'},
                ]
            }
        }
    
    # user defined matches
    columns_user = [
        {
            "name": "Dataframe_1",
            "id": "Dataframe_1",
            "presentation": "dropdown",
            "editable": True
        },
        {
            "name": "Dataframe_2",
            "id": "Dataframe_2",
            "presentation": "dropdown",
            "editable": True
        },
        {
            "name": "Matching_Options",
            "id": "Matching_Options",
            "presentation": "dropdown",
            "editable": True
        }
    ]

    if isinstance(df1, dict):
        df1 = pd.DataFrame(df1)
    if isinstance(df2, dict):
        df2 = pd.DataFrame(df2)
  
    dropdown_user = {
        'Dataframe_1': {
            'options': [{'label': col, 'value': col} for col in df1.columns]
        },
        'Dataframe_2': {
            'options': [{'label': col, 'value': col} for col in df2.columns]
        },
        'Matching_Options': {
            'options': [
                {'label': 'Unify', 'value': 'union'},
                {'label': 'Inverse And Unify', 'value': 'inverse'},
                {'label': 'Concat', 'value': 'row bind'}
            ]
        }
    }
    set_progress((str(100), str(100)))  # Progress complete

    return {'display' : 'block'}, columns, data, dropdown, columns_user, dropdown_user, {'display' : 'block'}




if __name__ == '__main__':
    uploader.register_callbacks()
    tabs.register_callbacks()
    app.run_server(debug=True, host='0.0.0.0', port=8080)
