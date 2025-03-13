import base64
import io

import dash
from dash import html, Output, no_update, Input, State, ctx, dcc
import dash_bootstrap_components as dbc
import zipfile
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, inspect
from sqlalchemy import text
import os
from dash import dash_table  # Import dash_table for DataTable
from .files_processing import process_sps_file, modify_var_names, clean_question, filter_dict_by_keys, \
    add_missing_instances, get_original_cols, n_ary_detector, extract_variables, calculate_embeddings, \
    calculate_sim_matrix_dtype, matching_hungarian, smaller_dataframe, bigger_dataframe, execute_integration, \
    sorted_neighborhood_g, fill_dup_matrix, transitive_closure, get_other_dups, resolve_dups, \
    remove_duplicates_and_update_df, merge_non_matched_columns

DATABASE_URL = "postgresql+psycopg2://postgres:root@postgres:5432/db"
UPLOAD_FOLDER_PATH = '/usr/app/src/uploaded_files/upload_zip'
data_store = {"first_dataset": None, "second_dataset": None}
alert = dbc.Alert(
    "Unallowed File Has Been Uploaded",
    id="alert-auto",
    is_open=False,
    color="danger",
    duration=4000,
)


def get_upload_component(id):
    return [
        html.Br(),
        alert,
        dcc.Upload(
            id='dcc_uploader_1',
            children=dbc.Button("Upload First Dataset", color="primary", className="mr-2"),
            multiple=False,  # Only allow one file per upload
            accept=".zip"  # Only allow .zip files
        ),
        html.Br(),
        dbc.Row([
            dbc.Col(html.Label("First Dataset Options:"))

        ]),
        dbc.Row([
            dbc.Col(html.Label("Does First Dataset Have Header?")),
            dbc.Col(dbc.Checklist(
                id='first-dataset-has-header',
                options=[{'label': 'Yes', 'value': 'yes'}],
                value=[], inline=True
            )),
        ]),
        html.Br(),
        dcc.Upload(
            id='dcc_uploader_2',
            children=dbc.Button("Upload Second Dataset", color="primary", className="mr-2"),
            multiple=False,  # Only allow one file per upload
            accept=".zip"  # Only allow .zip files
        ),
        html.Br(),
        dbc.Row([
            dbc.Col(html.Label("Second Dataset Options:"))

        ]),
        dbc.Row([
            dbc.Col(html.Label("Does Second Dataset Have Header?")),
            dbc.Col(dbc.Checklist(
                id='second-dataset-has-header',
                options=[{'label': 'Yes', 'value': 'yes'}],
                value=[], inline=True
            )),
        ]),
        html.Br(),
        dbc.Row([  # Keep the buttons for action
            dbc.Col(dbc.Button("Start Matching", id="start-matching-btn", color="primary")),
            dbc.Col(dbc.Button("Preprocess Data", id="preprocess-data-btn", color="secondary"))
        ]),
    ]


def register_callbacks() -> None:
    # Update the callback for preprocessing the data
    dash.callback(
        [
            Output('column-select-1', 'options'),
            Output('column-select-2', 'options'),
            Output('alert-auto', 'is_open'),
            Output('first-dataset-store', 'data'),
            Output('second-dataset-store', 'data'),
            Output('first-dataset-info-store', 'data'),
            Output('second-dataset-info-store', 'data'),
            Output('first-dataset-info-store-matching', 'data'),
            Output('second-dataset-info-store-matching', 'data'),
            Output('first-dataset-categorical-mappings', 'data'),
            Output('second-dataset-categorical-mappings', 'data'),

        ],
        Input('preprocess-data-btn', 'n_clicks'),
        State('dcc_uploader_1', 'contents'),
        State('dcc_uploader_1', 'filename'),
        State('dcc_uploader_2', 'contents'),
        State('dcc_uploader_2', 'filename'),
        State('first-dataset-has-header', 'value'),  # New state to check if first dataset has header
        State('second-dataset-has-header', 'value'),
        running=[(Output('modal-loading-preprocessing', 'is_open'), True, False)],
        prevent_initial_call=True
    )(_handle_preprocess_data)

    # Callback for updating column plots dynamically for the first dataset
    dash.callback(
        Output('dataset-1-dashboard', 'children'),
        Input('column-select-1', 'value'),
        State('first-dataset-store', 'data'),
        State('first-dataset-info-store', 'data'),
        prevent_initial_call=True
    )(_update_column_plot)

    # Callback for updating column plots dynamically for the second dataset
    dash.callback(
        Output('dataset-2-dashboard', 'children'),
        Input('column-select-2', 'value'),
        State('second-dataset-store', 'data'),
        State('second-dataset-info-store', 'data'),
        prevent_initial_call=True
    )(_update_column_plot)

    # export data callback
    dash.callback(
        Output('modal-alert-mismatch', 'is_open'),
        Output('modal-alert-body', 'children'),
        Output('download-matching-csv', 'data'),
        Output('download-labels-csv', 'data'),
        Output('download-instances-csv', 'data'),
        Output('download-non-matched-data-csv', 'data'),
        Output('download-non-matched-types-csv', 'data'),
        Output('datatable-cleaning', 'data'),
        Output('datatable-cleaning', 'columns'),
        Output('start-clean-btn', 'style'),
        Output('datatable-cleaning', 'tooltip_data'),
        Output('final-df-results', 'style'),
        Input('export-data-btn', 'n_clicks'),
        Input('with-er','value'),
        State('datatable-matching', 'data'),
        State('datatable-matching-user', 'data'),
        State('datatable-matching', 'selected_rows'),
        State('datatable-matching-user', 'selected_rows'),
        State('first-dataset-store', 'data'),
        State('first-dataset-info-store', 'data'),
        State('second-dataset-store', 'data'),
        State('second-dataset-info-store', 'data'),
        State('first-dataset-categorical-mappings', 'data'),
        State('second-dataset-categorical-mappings', 'data'),
        running=[(Output('modal-loading-exporting', 'is_open'), True, False)],
        prevent_inital_call=True
    )(_export_matched_data)


def _handle_preprocess_data(n_clicks, contents_1, filename_1, contents_2,
                            filename_2, header_option_1, header_option_2):
    if ctx.triggered_id:
        if contents_1 is None and contents_2 is None:
            return [html.Div("First dataset not uploaded."), html.Div("Second dataset not uploaded."), False, None,
                    None, None, None, None, None, None, None]

        if contents_1 is None:
            return [html.Div("First dataset not uploaded."), no_update, False, None, None, None, None, None, None, None,
                    None]

        if contents_2 is None:
            return [no_update, html.Div("Second dataset not uploaded."), False, None, None, None, None, None, None,
                    None, None]

        try:
            content_type_1, content_string_1 = contents_1.split(',')
            decoded_1 = base64.b64decode(content_string_1)

            content_type_2, content_string_2 = contents_2.split(',')
            decoded_2 = base64.b64decode(content_string_2)

            with zipfile.ZipFile(io.BytesIO(decoded_1), 'r') as zip_ref_1, zipfile.ZipFile(io.BytesIO(decoded_2),
                                                                                           'r') as zip_ref_2:
                for file_info in zip_ref_1.infolist():
                    file_name_1 = file_info.filename
                    with zip_ref_1.open(file_name_1) as file_1:
                        if file_name_1.endswith('.csv'):
                            if 'yes' in header_option_1:
                                df_1 = pd.read_csv(file_1)
                            else:
                                df_1 = pd.read_csv(file_1, header=None)

                        elif file_name_1.endswith('.sps'):
                            sps_content = file_1.read().decode('utf-8')
                            df_1.columns = extract_variables(sps_content)
                            sps_data_1, data_without_nary_1, categorical_mappings_1 = process_sps_file(sps_content,
                                                                                                       df_1)

                for file_info in zip_ref_2.infolist():
                    file_name_2 = file_info.filename
                    with zip_ref_2.open(file_name_2) as file_2:
                        if file_name_2.endswith('.csv'):
                            if 'yes' in header_option_2:
                                df_2 = pd.read_csv(file_2)
                            else:
                                df_2 = pd.read_csv(file_2, header=None)

                        elif file_name_2.endswith('.sps'):
                            sps_content = file_2.read().decode('utf-8')
                            df_2.columns = extract_variables(sps_content)
                            sps_data_2, data_without_nary_2, categorical_mappings_2 = process_sps_file(sps_content,
                                                                                                       df_2)

            new_info_1 = {}

            for dict_name, inner_dict in sps_data_1.items():
                new_inner_dict = {}
                for key, value in inner_dict.items():
                    keys = key.split(",")
                    for k in keys:
                        new_inner_dict[k.strip()] = value
                new_info_1[dict_name] = new_inner_dict

            new_info_2 = {}

            for dict_name, inner_dict in sps_data_2.items():
                new_inner_dict = {}
                for key, value in inner_dict.items():
                    keys = key.split(",")
                    for k in keys:
                        new_inner_dict[k.strip()] = value
                new_info_2[dict_name] = new_inner_dict

            # Update column options
            column_options_1 = [{'label': col, 'value': col} for col in data_without_nary_1.columns]
            column_options_2 = [{'label': col, 'value': col} for col in data_without_nary_2.columns]

            return [column_options_1, column_options_2, False, data_without_nary_1.to_dict(),
                    data_without_nary_2.to_dict(), new_info_1, new_info_2,
                    sps_data_1, sps_data_2, categorical_mappings_1, categorical_mappings_2]

        except Exception as e:
            print(str(e), flush=True)
            return [html.Div(f"An error occurred during preprocessing: {str(e)}"),
                    html.Div(f"An error occurred during preprocessing: {str(e)}"), True, None, None, None, None, None,
                    None, None, None]

    return [no_update, no_update, False, None, None, None, None, None, None, None, None]


def _update_column_plot(selected_column, dataset, info):
    if not selected_column or not dataset:
        return no_update

    df = pd.DataFrame(dataset)
    if selected_column not in df.columns:
        return no_update

    col_data = df[selected_column]
    col_type = info['variable_types'].get(selected_column, "unknown")  # Get column type from info
    question = info['questions'][selected_column]
    missing_values = col_data.isnull().sum()
    missing = f"Missing values: {missing_values}"
    nrows = len(df)

    if col_type == "SDATE10":
        temp_converted = pd.to_datetime(col_data, format='%Y-%m-%d', errors='coerce')
        col_data = temp_converted.dropna()

        if col_data.empty:
            return html.Div(f"No valid dates found in {selected_column}")

        fig = px.histogram(col_data, title=f"Date Distribution of {selected_column}",
                           nbins=50, opacity=0.7)

        stats_data = [
            {"Statistic": "Date Range",
             "Value": f"{col_data.min().strftime('%Y-%m-%d')} - {col_data.max().strftime('%Y-%m-%d')}"},
            {"Statistic": "Total Rows", "Value": nrows},
            {"Statistic": "Missing Values", "Value": missing_values}
        ]

    elif col_type == "F8":
        if col_data.dropna().empty:
            return html.Div(f"No numeric data found in {selected_column}")

        mean = col_data.mean()
        median = col_data.median()
        std_dev = col_data.std()
        min_val = col_data.min()
        max_val = col_data.max()
        skewness = col_data.skew()
        kurtosis = col_data.kurt()

        q25 = col_data.quantile(0.25)
        q50 = col_data.quantile(0.50)
        q75 = col_data.quantile(0.75)

        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]

        fig = px.histogram(col_data.dropna(), nbins=50, title=f"Distribution of {selected_column}",
                           opacity=0.7)
        fig.update_xaxes(range=[min_val, max_val])

        stats_data = [
            {"Statistic": "Mean", "Value": f"{mean:.2f}"},
            {"Statistic": "Median", "Value": f"{median:.2f}"},
            {"Statistic": "Min", "Value": f"{min_val:.2f}"},
            {"Statistic": "Max", "Value": f"{max_val:.2f}"},
            {"Statistic": "Std Dev", "Value": f"{std_dev:.2f}"},
            {"Statistic": "Skewness", "Value": f"{skewness:.2f}"},
            {"Statistic": "Kurtosis", "Value": f"{kurtosis:.2f}"},
            {"Statistic": "25th Percentile", "Value": f"{q25:.2f}"},
            {"Statistic": "50th Percentile (Median)", "Value": f"{q50:.2f}"},
            {"Statistic": "75th Percentile", "Value": f"{q75:.2f}"},
            {"Statistic": "Outliers Detected", "Value": f"{len(outliers)}"},
            {"Statistic": "Total Rows", "Value": nrows},
            {"Statistic": "Missing Values", "Value": missing_values}
        ]

    elif col_type == "F3":
        category_counts = col_data.value_counts()
        fig = px.bar(category_counts, title=f"Category Distribution of {selected_column}")

        stats_data = [
            {"Statistic": "Unique Categories", "Value": col_data.nunique()},
            {"Statistic": "Most Frequent Category", "Value": col_data.mode()[0]},
            {"Statistic": "Total Categories", "Value": len(category_counts)},
            {"Statistic": "Total Rows", "Value": nrows},
            {"Statistic": "Missing Values", "Value": missing_values}
        ]
    elif col_type == "A1000":
        col_data = col_data.dropna().astype(str)
        text_lengths = col_data.str.len()
        word_counts = col_data.str.split().apply(len)

        fig = px.histogram(text_lengths, title=f"Text Length Distribution in {selected_column}",
                           nbins=30, opacity=0.7)

        most_frequent = col_data.mode()[0] if not col_data.mode().empty else "None"
        longest_entry = col_data[col_data.str.len().idxmax()] if not col_data.dropna().empty else "None"

        stats_data = [
            {"Statistic": "Total Unique Entries", "Value": col_data.nunique()},
            {"Statistic": "Most Frequent Entry", "Value": most_frequent},
            {"Statistic": "Longest Entry (by length)", "Value": longest_entry},
            {"Statistic": "Mean Text Length", "Value": f"{text_lengths.mean():.2f}"},
            {"Statistic": "Median Text Length", "Value": f"{text_lengths.median():.2f}"},
            {"Statistic": "Total Rows", "Value": nrows},
            {"Statistic": "Missing Values", "Value": missing_values}
        ]
    else:
        return html.Div(f"Unsupported column type '{col_type}' for {selected_column}")

    # Create a smaller DataTable for displaying statistics
    stats_table = dash_table.DataTable(
        columns=[{"name": "Statistic", "id": "Statistic"}, {"name": "Value", "id": "Value"}],
        data=stats_data,
        style_table={'width': '50%', 'margin': 'auto'},  # Reduced width to 50%
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontWeight': 'bold',
            'fontSize': '12px'  # Smaller font size for header
        },
        style_cell={
            'textAlign': 'left',
            'padding': '5px',  # Smaller padding
            'fontSize': '12px',  # Smaller font size for cells
            'height': 'auto',  # Allows for compact row height
            'whiteSpace': 'normal'
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
        }
    )

    # Combine everything into a layout
    return html.Div([
        html.H5(f"{selected_column} - {missing}"),
        html.H5(f"{selected_column} - {question}"),
        stats_table,
        dcc.Graph(figure=fig)
    ])


def _export_matched_data(nc,er, data, data_user, selected_rows, selected_rows_user, df1, info1, df2, info2, cmap1, cmap2):
    if ctx.triggered_id == 'export-data-btn':
        data_df = pd.DataFrame(data)
        data_user_df = pd.DataFrame(data_user)
        if selected_rows:
            selected_rows_df = data_df.loc[selected_rows]
        else:
            selected_rows_df = pd.DataFrame()
        if selected_rows_user:
            selected_rows_user_df = data_user_df.loc[selected_rows_user]
        else:
            selected_rows_user_df = pd.DataFrame()
        df1 = pd.DataFrame(df1)
        df2 = pd.DataFrame(df2)
        smaller_df, smaller_info, cmap_smaller = smaller_dataframe(df1, info1, cmap1, df2, info2, cmap2)
        bigger_df, bigger_info, cmap_bigger = bigger_dataframe(df1, info1, cmap1, df2, info2, cmap2)

        df1.columns = [col.strip() for col in df1.columns]
        df2.columns = [col.strip() for col in df2.columns]
        if smaller_df is None and bigger_df is None:
            for col in df1.columns:
                if info1['variable_types'][col] == 'F3':
                    df1[col] = df1[col].fillna(-1).astype(int).astype(str)
                    df1[col] = df1[col].map(cmap1[col])

            for col in df2.columns:
                if info2['variable_types'][col] == 'F3':
                    df2[col] = df2[col].fillna(-1).astype(int).astype(str)
                    df2[col] = df2[col].map(cmap2[col])

            try:
                if selected_rows_df.empty:
                    integrated_df = pd.DataFrame()  # Assign an empty DataFrame
                else:
                    integrated_df, labels_auto, instances_auto = execute_integration(df1, info1, cmap1, df2, info2,
                                                                                     cmap2, selected_rows_df)

                if selected_rows_user_df.empty:
                    integrated_df_user = pd.DataFrame()
                else:
                    integrated_df_user, labels_user, instances_user = execute_integration(df1, info1, cmap1, df2, info2,
                                                                                          cmap2, selected_rows_user_df)
                    labels_auto = {k: [v] for k, v in labels_auto.items()}
                    labels_user = {k: [v] for k, v in labels_user.items()}
                    labels_auto_df = pd.DataFrame(labels_auto)
                    labels_user_df = pd.DataFrame(labels_user)
                    labels_concat = pd.concat([labels_auto_df, labels_user_df], axis=1)

                    instances_auto = {k: [v] for k, v in instances_auto.items()}
                    instances_user = {k: [v] for k, v in instances_user.items()}
                    instances_auto_df = pd.DataFrame(instances_auto)
                    instances_user_df = pd.DataFrame(instances_user)
                    instances_concat = pd.concat([instances_auto_df, instances_user_df], axis=1)
                    instances_concat.columns = [
                        f'{col}.{i}' if list(instances_concat.columns).count(col) > 1 and i > 0 else col
                        for i, col in enumerate(instances_concat.columns)]

                    combined_integrated_df = pd.concat([integrated_df, integrated_df_user], axis=1)
                    source_column = ['Dataframe_1'] * len(df1) + ['Dataframe_2'] * len(df2)
                    combined_integrated_df['source'] = source_column
                    non_matched_df, non_matched_types = merge_non_matched_columns(df1, info1, df2
                                                                                  , info2, selected_rows_df,
                                                                                  selected_rows_user_df)
                    deduped_df = combined_integrated_df.drop_duplicates(
                        subset=combined_integrated_df.columns[combined_integrated_df.columns != 'source'])
                    tuples_dup, df_sorted = sorted_neighborhood_g(deduped_df, labels_concat)
                    final_df = resolve_dups(df_sorted, get_other_dups(
                        transitive_closure(fill_dup_matrix(tuples_dup, len(df_sorted), df_sorted, labels_concat)),
                        tuples_dup))
                    df_to_output = final_df.copy().reset_index(drop=True)
                    labels_dict = {key: list(value.values())[0] for key, value in labels_concat.to_dict().items()}
                    print(labels_dict.keys(), flush=True)
                    print(df_to_output.columns, flush=True)
                    df_to_output, l = remove_duplicates_and_update_df(df_to_output, labels_dict)
                    df_to_output.insert(0, 'index', df_to_output.index)
                    engine = create_engine(DATABASE_URL)
                    df_to_output.to_sql('integrated_table', engine, if_exists='replace', index=False)
                    tootltip = [
                        {
                            column: {'value': str(value), 'type': 'markdown'}
                            for column, value in row.items()
                        } for row in df_to_output.to_dict('records')
                    ]
                return (no_update, no_update, dcc.send_data_frame(final_df.to_csv, "Fragebogen_integrated.csv")
                        , dcc.send_data_frame(labels_concat.to_csv, "labels_type_integrated.csv")
                        , dcc.send_data_frame(instances_concat.to_csv, "labels_instances_integrated.csv")
                        , dcc.send_data_frame(non_matched_df.to_csv, "non_matched_data.csv")
                        , dcc.send_data_frame(non_matched_types.to_csv, "non_matched_types.csv")
                        , df_to_output.to_dict('records'), [{"name": i, "id": i} for i in df_to_output.columns],
                        {'display': 'block'}, tootltip, {'display': 'block'})
            except Exception as e:
                print(f"Error: {e}", flush=True)
                return True, 'Non Compatible Types were chosen', no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        else:
            for col in smaller_df.columns:
                if smaller_info['variable_types'][col] == 'F3':
                    smaller_df[col] = smaller_df[col].fillna(-1).astype(int).astype(str)
                    smaller_df[col] = smaller_df[col].map(cmap_smaller[col])
            for col in bigger_df.columns:
                if bigger_info['variable_types'][col] == 'F3':
                    bigger_df[col] = bigger_df[col].fillna(-1).astype(int).astype(str)
                    bigger_df[col] = bigger_df[col].map(cmap_bigger[col])
            try:
                # merging
                integrated_df, labels_auto, instances_auto = execute_integration(smaller_df, smaller_info, cmap_smaller,
                                                                                 bigger_df
                                                                                 , bigger_info, cmap_bigger,
                                                                                 selected_rows_df)

                integrated_df_user, labels_user, instances_user = execute_integration(smaller_df, smaller_info,
                                                                                      cmap_smaller, bigger_df
                                                                                      , bigger_info, cmap_bigger,
                                                                                      selected_rows_user_df)
                labels_auto = {k: [v] for k, v in labels_auto.items()}
                labels_user = {k: [v] for k, v in labels_user.items()}
                labels_auto_df = pd.DataFrame(labels_auto)
                labels_user_df = pd.DataFrame(labels_user)
                labels_concat = pd.concat([labels_auto_df, labels_user_df], axis=1)
                labels_concat.columns = [f'{col}.{i}' if list(labels_concat.columns).count(col) > 1 and i > 0 else col
                                         for i, col in enumerate(labels_concat.columns)]

                instances_auto = {k: [v] for k, v in instances_auto.items()}
                instances_user = {k: [v] for k, v in instances_user.items()}
                instances_auto_df = pd.DataFrame(instances_auto)
                instances_user_df = pd.DataFrame(instances_user)
                instances_concat = pd.concat([instances_auto_df, instances_user_df], axis=1)
                difference_instances = labels_concat.shape[1] - instances_concat.shape[1]
                instances_concat.columns = [
                    f'{col}.{i + difference_instances - 1}' if list(instances_concat.columns).count(
                        col) > 1 and i > 0 else col
                    for i, col in enumerate(instances_concat.columns)]

                combined_integrated_df = pd.concat([integrated_df, integrated_df_user], axis=1)
                combined_integrated_df.columns = [
                    f'{col}.{i}' if list(combined_integrated_df.columns).count(col) > 1 and i > 0 else col
                    for i, col in enumerate(combined_integrated_df.columns)]
                source_column = ['Dataframe_1'] * len(df1) + ['Dataframe_2'] * len(df2)

                combined_integrated_df['source'] = source_column
                # add non matched columns as dataset
                non_matched_df, non_matched_types = merge_non_matched_columns(smaller_df, smaller_info, bigger_df
                                                                              , bigger_info, selected_rows_df,
                                                                              selected_rows_user_df)
                union_df = pd.concat([combined_integrated_df.reset_index(drop=True), non_matched_df.reset_index(drop=True)], axis=1)
                # if er checked do er if not return deduped_df

                print('Before ER : ', len(combined_integrated_df), flush=True)
                deduped_df = combined_integrated_df.drop_duplicates(
                    subset=combined_integrated_df.columns[combined_integrated_df.columns != 'source'])
                if not er:
                    labels_dict = {key: list(value.values())[0] for key, value in labels_concat.to_dict().items()}
                    df_to_output, l = remove_duplicates_and_update_df(deduped_df, labels_dict)
                    df_to_output.insert(0, 'index', df_to_output.index)
                    engine = create_engine(DATABASE_URL)
                    df_to_output.to_sql('integrated_table', engine, if_exists='replace', index=False)
                    tootltip = [
                        {
                            column: {'value': str(value), 'type': 'markdown'}
                            for column, value in row.items()
                        } for row in deduped_df.to_dict('records')
                    ]
                    return (no_update, no_update, dcc.send_data_frame(deduped_df.to_csv, "Fragebogen_integrated.csv")
                            , dcc.send_data_frame(labels_concat.to_csv, "labels_type_integrated.csv")
                            , dcc.send_data_frame(instances_concat.to_csv, "labels_instances_integrated.csv")
                            , dcc.send_data_frame(union_df.to_csv, "union_df.csv")
                            , dcc.send_data_frame(non_matched_types.to_csv, "non_matched_types.csv")
                            , deduped_df.to_dict('records'), [{"name": i, "id": i} for i in deduped_df.columns],
                            {'display': 'block'}, tootltip, {'display': 'block'})
                tuples_dup, df_sorted = sorted_neighborhood_g(deduped_df, labels_concat)
                final_df = resolve_dups(df_sorted,tuples_dup)
                print('finished here')
                # uncomment to run transitive closure
                #final_df = resolve_dups(df_sorted, get_other_dups(
                #    transitive_closure(fill_dup_matrix(tuples_dup, len(df_sorted))),
                #    tuples_dup))
                print('After ER : ', len(final_df), flush=True)
                df_to_output = final_df.copy().reset_index(drop=True)
                labels_dict = {key: list(value.values())[0] for key, value in labels_concat.to_dict().items()}
                df_to_output, l = remove_duplicates_and_update_df(df_to_output, labels_dict)
                df_to_output.insert(0, 'index', df_to_output.index)
                engine = create_engine(DATABASE_URL)
                df_to_output.to_sql('integrated_table', engine, if_exists='replace', index=False)
                tootltip = [
                    {
                        column: {'value': str(value), 'type': 'markdown'}
                        for column, value in row.items()
                    } for row in df_to_output.to_dict('records')
                ]
                return (no_update, no_update, dcc.send_data_frame(df_to_output.to_csv, "Fragebogen_integrated.csv")
                        , dcc.send_data_frame(labels_concat.to_csv, "labels_type_integrated.csv")
                        , dcc.send_data_frame(instances_concat.to_csv, "labels_instances_integrated.csv")
                        , dcc.send_data_frame(union_df.to_csv, "union_df.csv")
                        , dcc.send_data_frame(non_matched_types.to_csv, "non_matched_types.csv")
                        , df_to_output.to_dict('records'), [{"name": i, "id": i} for i in df_to_output.columns],
                        {'display': 'block'}, tootltip, {'display': 'block'})

            except Exception as e:
                print(f"Error: {e}", flush=True)
                return True, 'Non Compatible Types were chosen', no_update, no_update, no_update,no_update, no_update, no_update, no_update, no_update, no_update, no_update
    return no_update, no_update, no_update, no_update, no_update,no_update, no_update, no_update, no_update, no_update, no_update, no_update
