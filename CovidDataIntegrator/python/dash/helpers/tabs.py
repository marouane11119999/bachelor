import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, callback, Input, State, Output, ctx, ALL
import dash
from .uploader import get_upload_component
from sqlalchemy import create_engine, inspect, text
import base64
import pandas as pd
import io
import requests
from .files_processing import errors_nan, errors_zero, check_valid_dates_in_column, detect_error_textual

DATABASE_URL = "postgresql+psycopg2://postgres:root@postgres:5432/db"


def get_integration_columns():
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    columns = inspector.get_columns('integrated_table')
    return columns


def custom_error():
    alert_errors = dbc.Alert(
        "Error Uploading Data",
        id="alert-auto-csv-errors",
        is_open=False,
        color="danger",
        duration=4000,
    )
    return [
        html.Br(),
        html.H4("Upload your errors as a .csv, following this format: (row number, column number, error):"),
        html.Br(),
        alert_errors,
        dcc.Upload(
            id='dcc_uploader_errors_csv',
            children=dbc.Button("Upload Errors", color="primary", className="mr-2"),
            multiple=False,  # Only allow one file per upload
            accept=".csv"  # Only allow .csv files
        ),
        html.Br(),
        html.Div(id="output-error-csv")
    ]


def custom_correction():
    alert_correction = dbc.Alert(
        "Error Uploading Data",
        id="alert-auto-csv-correction",
        is_open=False,
        color="danger",
        duration=4000,
    )
    return [
        html.Br(),
        html.H3(
            "Upload your correction suggestions as a .csv, following this format: (row number, column number, correction):"),
        html.Br(),
        alert_correction,
        dcc.Upload(
            id='dcc_uploader_correction_csv',
            children=dbc.Button("Upload Corrections", color="primary", className="mr-2"),
            multiple=False,  # Only allow one file per upload
            accept=".csv"  # Only allow .csv files
        ),
        html.Br(),
        dbc.Button("Call Mimir", id='call-mimir', color="primary", className="mr-2")
    ]


alert_cleaning = dbc.Alert(
    "No Errors/No Corrections",
    id="alert-uploads-correrror",
    is_open=False,
    color="danger",
    duration=4000,
)

modal_loading_cleaning = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Cleaning..."), close_button=False),
        dbc.ModalBody(children=["Mimir is Cleaning...   ", dbc.Spinner(color="dark")]),
    ],
    id="modal-loading-cleaning",
    centered=True,
    is_open=False,
)


def get_tabs():
    return html.Div(
        [
            dbc.Tabs(
                [
                    # Data Upload Tab
                    dbc.Tab(
                        children=html.Div(get_upload_component('dash_uploader')),
                        label="Data Upload",
                        active_tab_style={"textTransform": "uppercase"},
                        active_label_style={"color": "#FB79B3"}
                    ),

                    # Summaries Tab for the First Dataset
                    dbc.Tab(
                        children=html.Div(id='summaries-dataset-1', children=[
                            html.Br(),
                            html.Label("Select Column for Dataset 1:"),
                            dcc.Dropdown(id='column-select-1', options=[], value=None),
                            html.Div(id='dataset-1-summary'),
                            html.Br(),
                            html.Div(id='dataset-1-dashboard')  # Dashboard for dataset 1
                        ]),
                        label="Summaries (Dataset 1)",
                        active_tab_style={"textTransform": "uppercase"},
                        active_label_style={"color": "#FB79B3"}
                    ),

                    # Summaries Tab for the Second Dataset
                    dbc.Tab(
                        children=html.Div(id='summaries-dataset-2', children=[
                            html.Label("Select Column for Dataset 2:"),
                            dcc.Dropdown(id='column-select-2', options=[], value=None),
                            html.Div(id='dataset-2-summary'),
                            html.Br(),
                            html.Div(id='dataset-2-dashboard')  # Dashboard for dataset 2
                        ]),
                        label="Summaries (Dataset 2)",
                        active_tab_style={"textTransform": "uppercase"},
                        active_label_style={"color": "#FB79B3"}
                    ),

                    # Matching Results Tab
                    dbc.Tab(
                        children=html.Div(id='matching-results',
                                          children=[
                                              html.Br(),
                                              html.H3("Matching Results"),
                                              html.Div(id='matching-result'
                                                       , children=[
                                                      dash_table.DataTable(
                                                          id='datatable-matching',
                                                          columns=[],
                                                          data=[],
                                                          style_cell={'textAlign': 'left'},
                                                          row_selectable="multi",
                                                          dropdown={},
                                                          style_table={"overflowX": "auto"},
                                                      ),
                                                      html.Br(),
                                                      html.H3('User Defined Matches : '),
                                                      html.Br(),
                                                      dash_table.DataTable(
                                                          id='datatable-matching-user',
                                                          columns=[],
                                                          data=[],
                                                          style_cell={'textAlign': 'left'},
                                                          row_selectable="multi",
                                                          dropdown={},
                                                          style_table={"overflowX": "auto", 'overflowY': 'auto'},
                                                      ),
                                                      html.Br(),
                                                      dbc.Button("Add Row", id="add-data-btn", color="secondary",
                                                                 style={'display': 'none'})
                                                  ]
                                                       , className="dbc dbc-row-selectable", ),
                                              html.Br(),
                                              html.Br(),
                                              dcc.Download(id="download-matching-csv"),
                                              dcc.Download(id="download-labels-csv"),
                                              dcc.Download(id="download-instances-csv"),
                                              dcc.Download(id="download-non-matched-data-csv"),
                                              dcc.Download(id="download-non-matched-types-csv"),
                                              dbc.Checkbox(
                                                  id="with-er",
                                                  label="Resolve duplicates",
                                                  value=False,
                                              ),
                                              dbc.Button("Export Data", id="export-data-btn", color="secondary",
                                                         style={'display': 'none'})
                                          ]),
                        label="Matching Results",
                        active_tab_style={"textTransform": "uppercase"},
                        active_label_style={"color": "#FB79B3"}
                    ),
                    # Result Dataset / error detection / cleaning
                    dbc.Tab(
                        children=html.Div(id='final-df-results',
                                          children=[
                                              html.Br(),
                                              html.H3("Cleaning"),
                                              html.Div(id='cleaning-result',
                                                       children=[
                                                           dash_table.DataTable(
                                                               id='datatable-cleaning',
                                                               columns=[],
                                                               data=[],
                                                               style_table={
                                                                   'overflowX': 'auto',
                                                                   'height': '300px',
                                                                   'overflowY': 'auto'
                                                               },
                                                               style_cell={
                                                                   'overflow': 'hidden',
                                                                   'textOverflow': 'ellipsis',
                                                                   'maxWidth': 0,
                                                               },
                                                               tooltip_duration=None
                                                           ),
                                                           dcc.Download(id="download-clean-csv"),
                                                           alert_cleaning,
                                                           html.Div(id='error-detection-checks', children=[]),
                                                           modal_loading_cleaning,
                                                           html.Div(id='custom-error-input', children=custom_error()),
                                                           html.Br(),
                                                           dbc.Label("Labeling Budget"),
                                                           dbc.Input(id="input-labeling",
                                                                     placeholder="How many errors you can correct for Mimir?",
                                                                     type="number"),
                                                           dbc.Button("Save Errors", id="start-clean-btn",
                                                                      color="secondary", style={'display': 'block'}),
                                                           html.Br(),
                                                           html.Div(id='custom-correction-input',
                                                                    children=custom_correction(),
                                                                    style={'display': 'block'})
                                                       ]
                                                       )
                                          ]),

                        label="Cleaning",
                        active_tab_style={"textTransform": "uppercase"},
                        active_label_style={"color": "#FB79B3"}
                    )
                ]
            )
        ]
    )


def register_callbacks():
    dash.callback(
        Output('datatable-matching-user', 'data'),
        Input('add-data-btn', 'n_clicks'),
        State('datatable-matching-user', 'data'),
        State('datatable-matching-user', 'columns'),
        prevent_intial_call=True
    )(_add_user_defined_row)
    dash.callback(
        Output('error-detection-checks', 'children', allow_duplicate=True),
        Input('datatable-cleaning', 'data'),
        prevent_initial_call=True
    )(_output_error_checks)
    dash.callback(
        Output('output-error-csv', 'children'),
        Output('alert-uploads-correrror', 'is_open'),
        Output('custom-correction-input', 'style'),
        # Output('download-clean-csv', 'data'),
        Input('dcc_uploader_errors_csv', 'contents'),
        # Input('dcc_uploader_correction_csv', 'contents'),
        Input('start-clean-btn', 'n_clicks'),
        State('input-labeling', 'value'),
        State({"type": "checklist", "index": ALL}, "value"),
        prevent_initial_call=True
    )(_execute_error_detection)
    dash.callback(
        Output('output-error-csv', 'children', allow_duplicate=True),
        Output('download-clean-csv', 'data'),
        Input('call-mimir', 'n_clicks'),
        Input('dcc_uploader_correction_csv', 'contents'),
        State('input-labeling', 'value'),
        prevent_initial_call=True
    )(_clean_data)


def _output_error_checks(data):
    if data is not None:
        columns = list(data[0].keys())
        checkboxes = []

        for col in columns:
            checkboxes.append(
                html.Div(
                    [html.Br(),
                     html.Label(f'Error checks for {col}:'),
                     html.Br(),
                     dbc.Checklist(
                         id={"type": "checklist", "index": col},
                         options=[
                             {'label': 'Check NaNs', 'value': 'check_nans'},
                             {'label': 'Check Zero Values', 'value': 'check_zero'},
                             {'label': 'Spellcheck', 'value': 'check_spelling'},
                             {'label': 'Date Validity', 'value': 'check_date'},

                         ],
                         inline=True,
                         value=[]
                     )
                     ]
                )
            )
        return checkboxes
    return dash.no_update


def _add_user_defined_row(n_clicks, rows, columns):
    if ctx.triggered_id == 'add-data-btn':
        rows.append({c['id']: '' for c in columns})
    return rows


def _execute_error_detection(content, nc, lab, values):
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM integrated_table"
    errors_dict = {}
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    if ctx.triggered_id == 'start-clean-btn':
        # if content_mimir is None:
        #    return 'No correction suggested', False,dash.no_update
        for i, check in enumerate(values):
            for ch in check:
                if len(ch) == 0:
                    continue
                elif ch == 'check_nans':
                    print('checking nans col')
                    errors_dict.update(errors_nan(df, df.columns[i]))
                elif ch == 'check_zero':
                    errors_dict.update(errors_zero(df, df.columns[i]))
                elif ch == 'check_spelling':
                    errors_dict.update(detect_error_textual(df, df.columns[i]))
                elif ch == 'check_date':
                    errors_dict.update(check_valid_dates_in_column(df, df.columns[i]))
        if content is not None:
            user_dict = {}
            ct, cs = content.split(',')
            decoded = base64.b64decode(cs)
            df_errors = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
            for index, row in df_errors.iterrows():
                e1, e2, e3 = row
                if pd.isna(e3):
                    user_dict[(e1, e2)] = 'N/A'
                else:
                    user_dict[(e1, e2)] = e3
            errors_dict.update(user_dict)
        errors_df = pd.DataFrame(
            [(key[0], key[1], value) for key, value in errors_dict.items()],
            columns=['row', 'cell', 'error']
        )
        engine = create_engine(DATABASE_URL)
        x = lab if lab is not None else 10
        error_counts = errors_df.groupby('row').size().reset_index(name='error_count')
        top_10_errors = error_counts.nlargest(x, 'error_count')
        top_errors = list(top_10_errors['row'])
        errors_df.to_sql('errors', engine, if_exists='replace', index=False)
        return 'Please correct the tuples : ' + str(top_errors), dash.no_update, {'display': 'block'}
        '''if content_mimir is not None:
            corrected_df = df.copy()
            ct, cs = content_mimir.split(',')
            decoded = base64.b64decode(cs)
            df_mimir = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
            for index, row in df_mimir.iterrows():
                e1, e2, e3 = row
                corrected_df.iloc[int(e1), int(e2)] = e3
            engine = create_engine(DATABASE_URL)
            corrected_df.to_sql('clean', engine, if_exists='replace', index=False)
            # ready for mimir here dirty:df,corrected_df:clean,errors:errors_dict,budget:len(df_mimir)
            try:
                # Send POST request to Mimir service
                mimir_url = "http://mimir:5001/process"  # Use 'mimir' as hostname
                response = requests.post(mimir_url, json={"budget": len(df_mimir)})

                if response.status_code == 200:
                    # Parse and display the result
                    result = response.json()
                    with engine.connect() as connection:
                        query = "SELECT * FROM cleanest_data"
                        cleanest = pd.read_sql(query, connection)
                        # drop all data after finishing cleaning : only errors and clean versions
                        query_errors = text("DROP TABLE errors")
                        query_clean = text("DROP TABLE clean")
                        query_cleanest = text("DROP TABLE cleanest_data")
                        query_corrections = text("DROP TABLE corrections")
                        connection.execute(query_errors)
                        connection.execute(query_clean)
                        connection.execute(query_cleanest)
                        connection.execute(query_corrections)
                        connection.commit()
                    return ("Processing Complete"
                            , dash.no_update
                            , dcc.send_data_frame(cleanest.to_csv, "cleanest.csv"))
                else:
                    return f"Error: {response.json().get('message', 'Unknown error')}", dash.no_update, dash.no_update
            except Exception as e:
                return f"Error connecting to Mimir: {str(e)}", dash.no_update, dash.no_update'''
    return dash.no_update, dash.no_update, dash.no_update


def _clean_data(nc, content_mimir,lab):
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM integrated_table"
    query_errors = "SELECT * FROM errors"
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
        errors = pd.read_sql(query_errors, connection)
    if ctx.triggered_id == 'call-mimir':
        if content_mimir is None:
            return 'No correction suggested', dash.no_update
        else:
            corrected_df = df.copy()
            ct, cs = content_mimir.split(',')
            decoded = base64.b64decode(cs)
            df_mimir = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
            for index, row in df_mimir.iterrows():
                e1, e2, e3 = row
                corrected_df.iloc[int(e1), int(e2)] = e3
            engine = create_engine(DATABASE_URL)
            corrected_df.to_sql('clean', engine, if_exists='replace', index=False)
            # ready for mimir here dirty:df,corrected_df:clean,errors:errors_dict,budget:len(df_mimir)
            try:
                # Send POST request to Mimir service
                mimir_url = "http://mimir:5001/process"  # Use 'mimir' as hostname
                response = requests.post(mimir_url, json={"budget": lab})

                if response.status_code == 200:
                    # Parse and display the result
                    result = response.json()
                    with engine.connect() as connection:
                        query = "SELECT * FROM cleanest_data"
                        cleanest = pd.read_sql(query, connection)
                        # drop all data after finishing cleaning : only errors and clean versions
                        query_errors = text("DROP TABLE errors")
                        query_clean = text("DROP TABLE clean")
                        query_cleanest = text("DROP TABLE cleanest_data")
                        query_corrections = text("DROP TABLE corrections")
                        connection.execute(query_errors)
                        connection.execute(query_clean)
                        connection.execute(query_cleanest)
                        connection.execute(query_corrections)
                        connection.commit()
                    return "Processing Complete", dcc.send_data_frame(cleanest.to_csv, "cleanest.csv")
                else:
                    return f"Error: {response.json().get('message', 'Unknown error')}", dash.no_update
            except Exception as e:
                return f"Error connecting to Mimir: {str(e)}", dash.no_update
    return dash.no_update,dash.no_update

