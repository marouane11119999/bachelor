import nltk
import string
import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util
import numpy as np
from munkres import Munkres
from nltk.corpus import stopwords
import re
from collections import defaultdict
from typing import Dict, Tuple, Any, Set
from datetime import datetime
from typing import Tuple, Optional
import uuid
import textdistance
import random
import csv
import phunspell
from collections import defaultdict
from Levenshtein import jaro_winkler
import pandas as pd

pspell = phunspell.Phunspell('de_DE')


def jaro_winkler_similarity(str1, str2):
    return jaro_winkler(str1, str2)


def find_high_jaro_winkler_pair(df):
    similarity_map = defaultdict(list)

    # Filter columns with only 0,1 as unique values
    binary_columns = [col for col in df.columns if set(df[col].unique()) <= {0, 1}]
    result_pairs = []
    for i in range(len(binary_columns)):
        for j in range(i + 1, len(binary_columns)):
            similarity = jaro_winkler_similarity(binary_columns[i],
                                                 binary_columns[j])
            if similarity > 0.90:
                similarity_map[binary_columns[i]].append(binary_columns[j])
                similarity_map[binary_columns[j]].append(binary_columns[i])

    for key, values in similarity_map.items():
        if len(values) == 1:
            result_pairs.append((key, values[0]))

    return result_pairs if result_pairs else None


def n_ary_detector(var_1, var_2, data):
    unique_values_var_1 = set(data[var_1].unique())
    unique_values_var_2 = set(data[var_2].unique())

    if unique_values_var_1 != {0, 1} or unique_values_var_2 != {0, 1}:
        return False
    for i in range(len(data)):
        if data.iloc[i][var_1] == data.iloc[i][var_2] == 1:
            return False
    return True


def get_original_cols(variables, radical):
    originals = []
    for v in variables:
        if radical in v:
            originals.append(v)
    return originals


def filter_dict_by_keys(original_dict, keys_to_keep):
    filtered_dict = {key: original_dict[key] for key in keys_to_keep if key in original_dict}
    return filtered_dict


def clean_question(modified_var_names):
    var_quest_modified = modified_var_names
    # removing columns we dont need since the second df deals only with after vaccine --> vor infektion und vor impfung
    filtered_var_quest = {}
    for key, value in var_quest_modified.items():
        if "vor_Infektion" not in key and "vor_Impfung" not in key:
            # Remove suffixes 'nach_Infektion' and 'nach_Impfung'
            clean_key = key.replace("_nach_Infektion", "").replace("_nach_Impfung", "")
            filtered_var_quest[clean_key] = value
    return filtered_var_quest


def modify_var_names(var_quest_700):
    # Create a dictionary to count occurrences of each value
    value_counts = {}
    for key, value in var_quest_700.items():
        value_counts[value] = value_counts.get(value, 0) + 1

    # Iterate over the dictionary and modify keys and values based on the rules
    new_var_quest_700 = {}
    occurrences = {}

    for key, value in var_quest_700.items():
        # Get the current count of this value's occurrences
        count = occurrences.get(value, 0) + 1
        occurrences[value] = count

        # Check if we need to modify the "Bitte wählen" values based on the key name
        if value == "bitte wählen":
            new_value = " ".join(key.split("_"))  # Modify value based on the key name
            new_key = key  # Keep the key unchanged
            new_var_quest_700[new_key] = new_value

        # Modify values occurring multiple times (2 or 4 times)
        elif value_counts[value] > 1:
            # Join all keys with the same value, separated by a comma
            new_key = ", ".join([k for k in var_quest_700 if var_quest_700[k] == value])
            new_var_quest_700[new_key] = value

        else:
            # If there are no modifications needed, keep the original key-value pair
            new_var_quest_700[key] = value

    return new_var_quest_700


def extract_variables(content):
    variables = extract_variable_names(content)
    return variables


def process_sps_file(content, data):
    variables = extract_variables(content)
    questions = dict(zip(variables, process_questions(extract_questions(content))))
    value_labels = extract_value_labels(content)
    instances = add_missing_instances(further_processing(value_labels),
                                      data)
    variables_types = extract_variables_with_types(content)
    categorical_mappings = get_mappings_categorical(value_labels)

    instances_without_nary, question_without_nary, variables_types_without_nary, data_without_nary, categorical_mappings = merge_n_ary(
        instances,
        variables,
        data,
        questions,
        variables_types, categorical_mappings)

    fin_dict = modify_and_group_dicts(question_without_nary, instances_without_nary,
                                      variables_types_without_nary)

    return fin_dict, data_without_nary, categorical_mappings


def modify_and_group_dicts(
        question_dict: Dict[str, str],
        instances: Dict[str, list],
        variable_types: Dict[str, str]
) -> Dict[str, Dict]:
    grouped_keys = defaultdict(list)
    for key, value in question_dict.items():
        if value.lower() == 'bitte wählen':
            value = re.sub(r'[^a-zA-Z0-9]', ' ', key)
        group_key = (
            value, frozenset(instances.get(key, [])), variable_types.get(key))
        grouped_keys[group_key].append(key)

    new_question_dict = {
        ", ".join(keys): question
        for (question, _, _), keys in grouped_keys.items()
    }

    new_instances = {
        ", ".join(keys): list(group_key[1])
        for group_key, keys in grouped_keys.items()
    }

    new_variable_types = {
        ", ".join(keys): group_key[2]
        for group_key, keys in grouped_keys.items()
    }

    result = {
        "questions": new_question_dict,
        "instances": new_instances,
        "variable_types": new_variable_types
    }
    return result


def string_intersection(str1, str2):
    return ''.join(
        [char for char in str1 if char in str2 and str2.count(char) > 0 and (str2 := str2.replace(char, '', 1))])


def merge_n_ary(instances: dict, variables: list, data: pd.DataFrame, questions: dict, var_types: dict, cmap: dict):
    # Create a copy of the instances dictionary to avoid modifying the original
    new_instances = instances.copy()
    new_questions = questions.copy()
    new_variable_types = var_types.copy()

    '''l = [v.split('_')[0] for v in new_instances.keys()]

    
    v_c = {}
    for e in set(l):
        v_c[e] = l.count(e)'''

    potential_columns = find_high_jaro_winkler_pair(data)

    data_copy = data.copy()
    cmap_copy = cmap.copy()

    if potential_columns:
        potential_columns = list(set(tuple(sorted(l)) for l in potential_columns))
        for originals in potential_columns:
            if n_ary_detector(originals[0], originals[1], data):
                k = string_intersection(originals[0], originals[1])
                new_instances[k] = new_instances.pop(originals[0])
                new_instances.pop(originals[1], None)
                new_questions[k] = new_questions.pop(originals[0])
                new_questions.pop(originals[1], None)
                new_variable_types[k] = new_variable_types.pop(originals[0])
                new_variable_types.pop(originals[1], None)
                data_copy[k] = data_copy[originals[0]] + data_copy[originals[1]]
                data_copy.drop([originals[0], originals[1]], axis=1, inplace=True)

                if originals[0] in cmap_copy.keys() and originals[1] in cmap_copy.keys():
                    cmap_copy[k] = cmap_copy.pop(originals[0])

                    cmap_copy.pop(originals[1], None)

    return new_instances, new_questions, new_variable_types, data_copy, cmap_copy


def extract_variables_with_types(content) -> dict:
    variables = {}

    # Match the block containing variable definitions
    pattern = re.compile(r'DATA LIST FREE\s+FILE.*?\/(.*?)\.', re.DOTALL)
    match = pattern.search(content)
    if match:
        variable_block = match.group(1)

        # Extract variable names and their types
        variable_entries = re.findall(r'(\w+)\s*\((.*?)\)', variable_block)
        for var_name, var_type in variable_entries:
            variables[var_name] = var_type.strip()  # Clean up type strings if needed

    return variables


def extract_value_labels(lines):
    lines = lines.splitlines()

    value_labels_section = False
    value_labels = {}
    current_variable = None
    first_value_label_line = False

    for line in lines:
        if 'VALUE LABELS' in line:
            value_labels_section = True
            first_value_label_line = True
            continue

        if value_labels_section and line.strip() == '':
            break

        if value_labels_section:
            line = line.strip()

            if line.startswith('/') or first_value_label_line:
                if current_variable:
                    value_labels[current_variable.split(" ")[0]] = ' '.join(current_values)

                if first_value_label_line:
                    current_variable = line
                else:
                    current_variable = line[1:]
                current_values = []
                first_value_label_line = False

            if current_variable and line:
                values = line.split()
                current_values.extend(values)

    if current_variable:
        value_labels[current_variable.split(" ")[0]] = ' '.join(current_values)
    return value_labels


def further_processing(instances: dict) -> dict:
    final_list = {k: v.split()[1:] for k, v in instances.items()}
    final_list = {k: ' '.join(v) for k, v in final_list.items()}

    final_list = {k: [ll.split("'")[i] for i in range(len(ll.split("'"))) if i % 2 != 0] for k, ll in
                  final_list.items()}
    final_list = {k: [liste[i] for i in range(len(liste)) if i % 2 != 0] for k, liste in final_list.items()}
    key_list = list(final_list.keys())
    for i in range(len(key_list)):
        if all(item == "" for item in final_list[key_list[i]]):
            final_list[key_list[i]] = final_list[key_list[i - 1]]
    return final_list


def add_missing_instances(instances: dict, df: pd.DataFrame) -> dict:
    if len(instances) != len(df.columns):
        # get difference
        difference = set(df.columns).difference(set(instances.keys()))
        for column in difference:
            unique_values = df[column].dropna().unique()
            instances[column] = list(unique_values)
    return instances


def extract_variable_names(sps_content) -> list:
    variable_names = []
    pattern = re.compile(r'DATA LIST FREE\s+FILE.*?\/(.*?)\.', re.DOTALL)
    match = pattern.search(sps_content)
    if match:
        variable_block = match.group(1)
        variable_names = re.findall(r'(\w+)\s*\(.*?\)', variable_block)
    return variable_names


def extract_questions(sps_content) -> list:
    questions = []

    # Use a regex pattern to find the VARIABLE LABEL section
    pattern = re.compile(r'VARIABLE LABEL\s+(.*?)\s+VALUE LABELS', re.DOTALL)
    match = pattern.search(sps_content)
    if match:
        variable_block = match.group(1)
        # Extract text between single quotes
        questions = re.findall(r"'(.*?)'", variable_block)

    def remove_eingeben_parentheses(text):
        return re.sub(r'\(.*?eingeben.*?\)', '', text).strip()

    cleaned_questions = [remove_eingeben_parentheses(text) for text in questions]
    return cleaned_questions


def process_questions(questions: list) -> list:
    german_stopwords = set(stopwords.words('german'))
    questions = [s.replace("'", "") for s in questions]
    filtered_questions = []
    for question in questions:
        word_list = nltk.word_tokenize(question.lower(), language='german')

        # Filter out non-alphabetic tokens and stopwords
        filtered_words = [word for word in word_list if
                          word not in string.punctuation]  # and word.lower() not in german_stopwords]
        # Join the filtered words back into a sentence (if desired)
        filtered_sentence = ' '.join(filtered_words)
        filtered_questions.append(filtered_sentence.encode('utf-8').decode('utf-8'))
        # filtered_questions.append(filtered_sentence)
    return filtered_questions


def load_embeddings():
    model = SentenceTransformer('svalabs/german-gpl-adapted-covid')
    return model


def calculate_embeddings(d1: dict, d2: dict, model):
    # Generate embeddings for values in d1 and d2
    embeddings_1 = model.encode(list(d1.values()))
    embeddings_2 = model.encode(list(d2.values()))

    # Map embeddings back to the original keys
    result_1 = {key: embedding for key, embedding in zip(d1.keys(), embeddings_1)}
    result_2 = {key: embedding for key, embedding in zip(d2.keys(), embeddings_2)}

    return result_1, result_2


def calculate_sim_matrix_dtype(list1: dict, list2: dict, info_dict1: dict, info_dict2: dict) -> pd.DataFrame:
    similarity_matrix_questions = np.zeros((len(list1), len(list2)))
    compatible_types = {('F8', 'F3'), ('F3', 'F8')}
    for idx_i, (key_i, emb1) in enumerate(list1.items()):
        for idx_j, (key_j, emb2) in enumerate(list2.items()):

            type1 = info_dict1['variable_types'].get(key_i)
            type2 = info_dict2['variable_types'].get(key_j)
            print('processing : ', key_i, ' ', key_j, ' ', type1, ' ', type2, flush=True)
            if (type1, type2) in compatible_types:

                similarity = util.cos_sim(emb1, emb2).item()
                similarity_matrix_questions[idx_i][idx_j] = max(similarity, 0)
            elif type1 == type2 == 'SDATE10':

                seconds1 = [sec for sec in convert_to_seconds(info_dict1['instances'].get(key_i)) if sec is not None]
                seconds2 = [sec for sec in convert_to_seconds(info_dict2['instances'].get(key_j)) if sec is not None]
                dmax = calculate_dmax(seconds1, seconds2)
                similarity_monge_elkan = monge_elkan_similarity(seconds1, seconds2, dmax)
                similarity_embeddings = util.cos_sim(emb1, emb2).item()
                similarity_matrix_questions[idx_i][idx_j] = 0.5 * similarity_monge_elkan + 0.5 * max(
                    similarity_embeddings, 0)
            elif type1 == type2 == 'F8':
                v1 = info_dict1['instances'].get(key_i)
                v2 = info_dict2['instances'].get(key_j)
                dmax = calculate_dmax(v1, v2)
                similarity_monge_elkan = monge_elkan_similarity(v1, v2, dmax)
                similarity_embeddings = util.cos_sim(emb1, emb2).item()
                similarity_matrix_questions[idx_i][idx_j] = 0.5 * similarity_monge_elkan + 0.5 * max(
                    similarity_embeddings, 0)

            elif type1 == 'A1000' and type2 == 'A1000':

                instances_e1 = info_dict1['instances'].get(key_i)

                instances_e2 = info_dict2['instances'].get(key_j)

                similarity_matrix = np.zeros((len(instances_e1), len(instances_e2)))

                if len(instances_e1) == 0 or len(instances_e2) == 0 or util.cos_sim(emb1, emb2).item() < 0.45:

                    similarity = 0

                else:

                    for idx_ii, instance_e1 in enumerate(instances_e1):

                        for idx_jj, instance_e2 in enumerate(instances_e2):
                            s1 = str(instance_e1)

                            s2 = str(instance_e2)

                            def jaro_winkler_similarity(s1, s2):
                                return textdistance.jaro_winkler(s1, s2)

                            similarity_matrix[idx_ii][idx_jj] = jaro_winkler_similarity(s1, s2)

                    max_sim_e1 = np.max(similarity_matrix, axis=1)

                    max_sim_e2 = np.max(similarity_matrix, axis=0)

                    similarity = (np.sum(max_sim_e1) + np.sum(max_sim_e2)) / (len(instances_e1) + len(instances_e2))

                question_similarity = util.cos_sim(emb1, emb2).item()

                similarity_matrix_questions[idx_i][idx_j] = 0.8 * similarity + 0.2 * max(question_similarity, 0)



            elif type1 == 'F3' and type2 == 'F3':

                instances_e1 = info_dict1['instances'].get(key_i)

                instances_e2 = info_dict2['instances'].get(key_j)

                similarity_matrix = np.zeros((len(instances_e1), len(instances_e2)))

                if len(instances_e1) == 0 or len(instances_e2) == 0:

                    similarity = 0

                else:

                    for idx_ii, instance_e1 in enumerate(instances_e1):

                        for idx_jj, instance_e2 in enumerate(instances_e2):
                            # Convert instances to strings if necessary

                            s1 = str(instance_e1)

                            s2 = str(instance_e2)

                            def jaro_winkler_similarity(s1, s2):
                                return textdistance.jaro_winkler(s1, s2)

                            similarity_matrix[idx_ii][idx_jj] = jaro_winkler_similarity(s1, s2)

                    max_sim_e1 = np.max(similarity_matrix, axis=1)

                    max_sim_e2 = np.max(similarity_matrix, axis=0)

                    similarity = (np.sum(max_sim_e1) + np.sum(max_sim_e2)) / (len(instances_e1) + len(instances_e2))

                question_similarity = util.cos_sim(emb1, emb2).item()

                # Adjust weighting for this case as needed

                if len(instances_e1) > 3 or len(instances_e2) > 3:
                    similarity_matrix_questions[idx_i][idx_j] = 0.3 * similarity + 0.7 * max(question_similarity, 0)
                else:
                    similarity_matrix_questions[idx_i][idx_j] = max(question_similarity, 0)

            else:
                # Set similarity to 0 if variable types don't match or are incompatible
                similarity_matrix_questions[idx_i][idx_j] = 0

    # Return the similarity matrix as a DataFrame
    return pd.DataFrame(similarity_matrix_questions, index=list1.keys(), columns=list2.keys())


def matching_hungarian(sim_matrix: pd.DataFrame) -> pd.DataFrame:
    # Step 0: Transform the similarity matrix into a cost matrix for the Munkres algorithm
    max_value = sim_matrix.values.max()
    cost_matrix = max_value - sim_matrix
    max_dim = max(cost_matrix.shape[0], cost_matrix.shape[1])

    # Padding the matrix with zeros to make it square if needed
    if max_dim > cost_matrix.shape[0]:  # Adding rows if necessary
        print("Adding rows")
        rows_to_add = pd.DataFrame(0, index=range(max_dim - cost_matrix.shape[0]), columns=cost_matrix.columns)
        cost_matrix = pd.concat([cost_matrix, rows_to_add], ignore_index=True)

    if max_dim > cost_matrix.shape[1]:  # Adding columns if necessary
        print("Adding columns")
        cols_to_add = pd.DataFrame(0, index=cost_matrix.index, columns=range(cost_matrix.shape[1], max_dim))
        cost_matrix = pd.concat([cost_matrix, cols_to_add], axis=1)

    # Step 1: Normalize rows and columns
    cost_matrix = cost_matrix.sub(cost_matrix.min(axis=1), axis=0)
    cost_matrix = cost_matrix.sub(cost_matrix.min(axis=0), axis=1)

    # Step 2: Compute Munkres (Hungarian) solution
    print("Calling Munkres")
    m = Munkres()
    indices = m.compute(cost_matrix.to_numpy())

    # Step 3: Create DataFrame with matches and similarity scores
    matches = []
    for row, col in indices:
        if row < sim_matrix.shape[0] and col < sim_matrix.shape[1]:  # Ignore padded cells
            proposer = sim_matrix.index[row]
            receiver = sim_matrix.columns[col]
            similarity = sim_matrix.iat[row, col]
            matches.append((proposer, receiver, similarity))

    matching_df = pd.DataFrame(matches, columns=['Dataframe_1', 'Dataframe_2', 'Similarity'])
    return matching_df


def convert_to_seconds(date_list):
    seconds_list = []

    for i, date_str in enumerate(date_list):
        try:
            # Parse the date string into a datetime object
            dt_object = datetime.strptime(date_str, "%Y-%m-%d")

            # Convert datetime object to seconds since the epoch
            seconds = int(dt_object.timestamp())
            seconds_list.append(seconds)
        except Exception as e:
            if i > 0:
                seconds_list.append(seconds_list[-1])
            else:
                # If the first element fails, append None or any placeholder value
                seconds_list.append(None)

    return seconds_list


def calculate_dmax(list1, list2):
    # Combine the values from both lists
    combined_values = list1 + list2

    # Calculate the domain bounds
    min_value = min(combined_values)
    max_value = max(combined_values)

    # Calculate dmax as the range
    dmax = max_value - min_value

    return dmax


def monge_elkan_similarity(list1, list2, dmax):
    # Convert lists to numpy arrays for efficient operations (optional, but can be faster for large lists)
    list1 = np.array(list1)
    list2 = np.array(list2)

    # Compute the Monge-Elkan similarity
    max_similarities = []
    for x in list1:
        max_similarity = max(simnum_abs(x, y, dmax) for y in list2)
        max_similarities.append(max_similarity)

    # Return the average of the maximum similarities
    return np.mean(max_similarities)


def simnum_abs(x, y, dmax):
    return 1 - abs(x - y) / dmax if dmax != 0 else 1


def further_processing_adapt(instances: dict) -> dict:
    final_list = {k: v.split()[1:] for k, v in instances.items()}
    final_list = {k: ' '.join(v) for k, v in final_list.items()}
    return final_list


def get_mappings_categorical(value_labels):
    categoricals = further_processing_adapt(value_labels)
    def create_mapping(input_string):
        # Regular expression to capture the number-value pairs (e.g., '1' 'Weiblich')
        pattern = r"'(\d+)' '([^']+)'"
        matches = re.findall(pattern, input_string)
        return {key: value for key, value in matches}

    def process_input(input_data):
        result = {}
        previous_value = None  # Variable to store the previous value when necessary
        for key, value in input_data.items():
            # Create the mapping for the current value
            current_mapping = create_mapping(value)

            if current_mapping:  # If the mapping is not empty
                result[key] = current_mapping
                previous_value = current_mapping  # Update previous_value with the new mapping
            else:
                if previous_value:  # If the mapping is empty, use the previous value
                    result[key] = previous_value

        return result

    return process_input(categoricals)


def smaller_dataframe(df1: pd.DataFrame, info1: dict, cmap1: dict, df2: pd.DataFrame, info2: dict, cmap2: dict) -> \
        Optional[Tuple[pd.DataFrame, dict, dict]]:
    if len(df1) == len(df2):
        return (None, None)
    return (df1, info1, cmap1) if len(df1) < len(df2) else (df2, info2, cmap2)


def bigger_dataframe(df1: pd.DataFrame, info1: dict, cmap1: dict, df2: pd.DataFrame, info2: dict, cmap2: dict) -> \
        Optional[Tuple[pd.DataFrame, dict, dict]]:
    if len(df1) == len(df2):
        return (None, None)
    return (df1, info1, cmap1) if len(df1) > len(df2) else (df2, info2, cmap2)


def merge_non_matched_columns(df1: pd.DataFrame, info1: dict, df2: pd.DataFrame, info2: dict,
                              selected_df: pd.DataFrame, selected_df_user: pd.DataFrame) -> Tuple[DataFrame, DataFrame]:
    df1_cols_selected = selected_df['Dataframe_1'].str.strip().tolist()
    df2_cols_selected = selected_df['Dataframe_2'].str.strip().tolist()
    df1_cols_user_selected = selected_df_user['Dataframe_1'].str.strip().tolist()
    df2_cols_user_selected = selected_df_user['Dataframe_2'].str.strip().tolist()
    not_integrated_df1 = [col for col in df1.columns if col not in
                          df1_cols_selected and col not in df1_cols_user_selected]
    not_integrated_df2 = [col for col in df2.columns if col not in
                          df2_cols_selected and col not in df2_cols_user_selected]

    df1_filtered = df1[not_integrated_df1]
    df2_filtered = df2[not_integrated_df2]
    merged_df = pd.concat([df1_filtered.reset_index(drop=True), df2_filtered.reset_index(drop=True)], axis=1)
    # get types
    info1_not_integrated = {col: info1['variable_types'].get(col, 'Unknown') for col in not_integrated_df1}
    info2_not_integrated = {col: info2['variable_types'].get(col, 'Unknown') for col in not_integrated_df2}
    merged_variable_types = pd.DataFrame(list(info1_not_integrated.items()) + list(info2_not_integrated.items()), columns=['Column', 'Variable_Type'])
    return merged_df, merged_variable_types


def unique_no_nan(x):
    return x.dropna().unique()
def execute_integration(df1: pd.DataFrame, info1: dict, cmap1: dict, df2: pd.DataFrame, info2: dict, cmap2: dict,
                        selected_df: pd.DataFrame) -> Tuple[DataFrame, Dict[Any, str], Dict[Any, Set[Any]]]:
    label_type = dict()
    label_instances = dict()
    # add case F8 == A1000
    compatible_types = {('F8', 'F3'), ('F3', 'F8'),
                        #('F8', 'A1000'), ('A1000', 'F8'),
                        }

    negation_map = {
        'Unchecked': 'Checked',
        'Checked': 'Unchecked'
    }
    integrated_df = pd.DataFrame()
    for index, row in selected_df.iterrows():
        col_df1 = row['Dataframe_1'].strip()
        col_df2 = row['Dataframe_2'].strip()
        match_option = row['Matching_Options']
        # case both F3 -> union or inverse of the categories
        if info1['variable_types'][col_df1] == info2['variable_types'][col_df2] == 'F3':
            if match_option == 'union':
                # Take the union of the categories
                df2[col_df2].rename(df1[col_df1].name)
                merged_categories = pd.concat([df1[col_df1], df2[col_df2]], ignore_index=True)
                integrated_df[col_df1] = merged_categories
                label_type[col_df1] = 'F3'
                label_instances[col_df1] = set(info1['instances'][col_df1]) | set(info2['instances'][col_df2])
            elif match_option == 'inverse':
                # Take the inverse (categories in df1 not in df2 and vice versa)
                unique_vals_df1 = unique_no_nan(df1[col_df1])
                unique_vals_df2 = unique_no_nan(df2[col_df2])
                if len(unique_vals_df1) == 2 and len(unique_vals_df2) == 2:
                    unique_vals1 = sorted(set(unique_vals_df1))
                    encoding_map1 = {val: idx for idx, val in enumerate(unique_vals1)}
                    unique_vals2 = sorted(set(unique_vals_df2))
                    encoding_map2 = {val: idx for idx, val in enumerate(unique_vals2)}
                    decoding_map = {idx: val for val, idx in encoding_map1.items()}
                    df1_encoded = df1[col_df1].map(encoding_map1)
                    df2_encoded = df2[col_df2].map(encoding_map2)
                    df2_negated = df2_encoded.map(lambda x: 1 - x if x in [0, 1] else x)
                    concatenated_values = pd.concat([df1_encoded, df2_negated], ignore_index=True)
                    restored_values = concatenated_values.map(decoding_map)
                    integrated_df[col_df1] = restored_values
                    label_type[col_df1] = 'F3'
                    label_instances[col_df1] = set(info1['instances'][col_df1]) | set(info2['instances'][col_df2])
                '''df2_col_values = df2[col_df2].apply(
                    lambda x: negation_map.get(x, x))  # Apply negation_map for 'Ja' <-> 'Nein'
                df2[col_df2].rename(df1[col_df1].name)
                concatenated_values = pd.concat([df1[col_df1], df2_col_values], ignore_index=True)
                integrated_df[col_df1] = concatenated_values
                label_type[col_df1] = 'F3'
                print(info1['instances'][col_df1], flush=True)
                label_instances[col_df1] = set(info1['instances'][col_df1]) | set(info2['instances'][col_df2])'''
            else:
                # Error case: Unknown matching option
                print(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}", flush=True)
                raise ValueError(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}")
        if (info1['variable_types'][col_df1], info2['variable_types'][col_df2]) in compatible_types:
            if match_option == 'row bind':
                if info1['variable_types'][col_df1] == 'F3':
                    reverse_map = {v: k for k, v in cmap1[col_df1].items()}
                    df1[col_df1] = df1[col_df1].map(reverse_map)
                if info2['variable_types'][col_df2] == 'F3':
                    reverse_map = {v: k for k, v in cmap2[col_df2].items()}
                    df2[col_df2] = df2[col_df2].map(reverse_map)

                df1[col_df1] = df1[col_df1].astype(str)
                df2[col_df2] = df2[col_df2].astype(str)
                df2[col_df2].rename(df1[col_df1].name)
                merged_values = pd.concat([df1[col_df1], df2[col_df2]], ignore_index=True)
                integrated_df[col_df1] = merged_values
                label_type[col_df1] = 'F8'
            else:
                # Error case: Unknown matching option
                print(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}", flush=True)
                raise ValueError(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}")

        if info1['variable_types'][col_df1] == info2['variable_types'][col_df2] == 'F8':
            if match_option == 'row bind':
                df2[col_df2].rename(df1[col_df1].name)
                merged_values = pd.concat([df1[col_df1], df2[col_df2]], ignore_index=True)
                integrated_df[col_df1] = merged_values
                label_type[col_df1] = 'F8'
            else:
                # Error case: Unknown matching option
                print(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}", flush=True)
                raise ValueError(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}")

        if info1['variable_types'][col_df1] == info2['variable_types'][col_df2] == 'A1000':
            if match_option == 'row bind':
                df2[col_df2].rename(df1[col_df1].name)
                merged_values = pd.concat([df1[col_df1], df2[col_df2]], ignore_index=True)
                integrated_df[col_df1] = merged_values
                label_type[col_df1] = 'A1000'
            else:
                # Error case: Unknown matching option
                print(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}", flush=True)
                raise ValueError(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}")

        if info1['variable_types'][col_df1] == info2['variable_types'][col_df2] == 'SDATE10':
            if match_option == 'row bind':
                df2[col_df2].rename(df1[col_df1].name)
                merged_values = pd.concat([df1[col_df1], df2[col_df2]], ignore_index=True)
                integrated_df[col_df1] = merged_values
                label_type[col_df1] = 'SDATE10'
            else:
                # Error case: Unknown matching option
                print(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}", flush=True)
                raise ValueError(f"Unknown matching option '{match_option}' for {col_df1} and {col_df2}")
    return integrated_df, label_type, label_instances


# duplicate detection section
def convert_to_seconds_str(date_str):
    try:
        # Parse the date string into a datetime object
        dt_object = datetime.strptime(date_str, "%Y-%m-%d")

        # Convert datetime object to seconds since the epoch
        seconds = int(dt_object.timestamp())
        return seconds
    except Exception as e:
        # Return None or an appropriate placeholder value for errors
        return 0


def generate_unique_value(row, length):
    # Check if the value is entirely 'nan' repeated to the specified length
    if row == ('nan' * length):
        # Generate a random UUID
        return str(uuid.uuid4())
    return row


def sorted_neighborhood_g(dataframe: pd.DataFrame, types: pd.DataFrame, target_efficiency=1, initial_window_size=3):
    types_dict = types.to_dict()
    types_dict = {key: list(value.values())[0] for key, value in types_dict.items()}
    blockers = [key for key in dataframe.columns]
    dataframe.loc[:,'blocker'] = (
        dataframe[blockers]
        .astype(str)
        .agg(lambda row: ''.join(val[0] for val in row), axis=1)  # Take first character
        .str.lower()
        .str.replace(' ', '', regex=True)
        .apply(generate_unique_value, args=(len(blockers),))
    )
    num_rows = len(dataframe)
    dataframe_sorted = dataframe.sort_values(by='blocker')
    duplicate_pairs = []
    comparisons = 0
    duplicates_found = 0
    window_size = initial_window_size
    for i in range(num_rows):

        window_end = min(i + window_size, num_rows)
        for j in range(i + 1, window_end):
            comparisons += 1
            comp = compare(dataframe_sorted, i, j, types)

            match = comp > 0.95
            #match = (dataframe_sorted.iloc[i]['blocker'] == dataframe_sorted.iloc[j]['blocker'])
            if match:
                duplicate_pairs.append((i, j))
                duplicates_found += 1
        if comparisons > 0:
            efficiency = duplicates_found / comparisons
            if efficiency > target_efficiency:
                window_size += 1
            else:
                window_size = initial_window_size
                comparisons = 0
                duplicates_found = 0
    dataframe_sorted.drop('blocker', axis=1, inplace=True)
    print('finished sn',flush=True)
    return duplicate_pairs, dataframe_sorted


def compare(df, index_i, index_j, types_dict):
    total_similarity = 0
    count = 0
    all_same_as_str = True
    for i in [col for col in df.columns if col != 'blocker' and col != 'source']:
        value_i = str(df.iloc[index_i][i])
        value_j = str(df.iloc[index_j][i])
        if value_i != value_j:
            all_same_as_str = False
            break
    if all_same_as_str:
        return 1.0
    for i in [col for col in df.columns if col != 'blocker' and col != 'source']:
        value_i = df.iloc[index_i][i]
        value_j = df.iloc[index_j][i]
        if pd.isna(value_i) and pd.isna(value_j):
            similarity = 0
        else:
            if types_dict.loc[0,i] == 'F3':
                similarity = 1 if value_i == value_j else 0
            elif types_dict.loc[0,i] == 'F8':
                if pd.isna(value_i) or pd.isna(value_j):

                    similarity = 0
                else:

                    col_floats = pd.to_numeric(df[i], errors='coerce').tolist()
                    value_i = float(value_i)
                    value_j = float(value_j)
                    min_val = min(col_floats)
                    max_val = max(col_floats)
                    d_max = max_val - min_val
                    if d_max == 0:

                        similarity = 1 if value_i == value_j else 0
                    else:

                        similarity = 1 - abs(value_i - value_j) / d_max
            elif types_dict.loc[0,i] == 'A1000':

                if pd.isna(value_i) or pd.isna(value_j):

                    similarity = 0
                else:

                    similarity = textdistance.jaro_winkler(str(value_i), str(value_j))
            else:

                if pd.isna(value_i) or pd.isna(value_j):
                    similarity = 0
                else:
                    col_seconds = convert_to_seconds(list(df[i].dropna()))
                    min_seconds = min(col_seconds)
                    max_seconds = max(col_seconds)
                    d_max = max_seconds - min_seconds
                    value_i_seconds = convert_to_seconds_str(value_i)
                    value_j_seconds = convert_to_seconds_str(value_j)
                    if d_max == 0:
                        similarity = 1 if value_i_seconds == value_j_seconds else 0
                    else:
                        similarity = 1 - abs(value_i_seconds - value_j_seconds) / d_max
            count += 1

        total_similarity += similarity
    return total_similarity / count if count > 0 else 0


def fill_dup_matrix(dup_tuples: list, length: int):
    dup_mat = np.eye(length)
    for i, j in dup_tuples:
        dup_mat[i, j] = 1
    return dup_mat


def transitive_closure_warshall(relation):
    n = len(relation)
    for i in range(n):
        for j in range(i + 1, n):
            if relation[i, j] == 1:
                for k in range(n):
                    if relation[j, k] == 1:
                        relation[i, k] = 1

    return relation


def transitive_closure(relation):
    n = len(relation)
    for i in range(1, n):
        for j in range(i):
            if relation[i, j] == 1:
                for k in range(n):
                    if relation[j, k] == 1:
                        relation[i, k] = 1
    for i in range(n - 1):
        for j in range(i + 1, n):  # j from i+1 to n
            if relation[i, j] == 1:
                for k in range(n):
                    if relation[j, k] == 1:
                        relation[i, k] = 1
    return relation


def get_other_dups(transitive_closure_matrix: np.ndarray, tuples_list: list) -> list:
    dup_list = tuples_list
    for i in range(len(transitive_closure_matrix)):
        for j in range(len(transitive_closure_matrix)):
            if i == j:
                continue
            elif transitive_closure_matrix[i, j] == 1 and (i, j) not in tuples_list:
                dup_list.append((i, j))
    return dup_list


def resolve_dups(df: pd.DataFrame, dups: list) -> pd.DataFrame:
    df['missing_values'] = df.isna().sum(axis=1)
    rows_to_drop = []

    for dup_pair in dups:
        row1, row2 = dup_pair

        if row1 not in df.index or row2 not in df.index:
            print(f"Skipping pair ({row1}, {row2}) as one of the rows does not exist.")
            continue
        missing_row1 = df.loc[row1, 'missing_values']
        missing_row2 = df.loc[row2, 'missing_values']

        if missing_row1 < missing_row2:
            rows_to_drop.append(row2)
        elif missing_row1 > missing_row2:
            rows_to_drop.append(row1)
        else:
            rows_to_drop.append(random.choice([row1, row2]))

    cleaned_df = df.drop(rows_to_drop, axis=0, errors='ignore')
    cleaned_df = cleaned_df.drop(columns=['missing_values'])
    return cleaned_df


def clean_string(input_string):
    cleaned_string = re.sub(r'[^\w\s]', ' ', input_string)
    cleaned_string = ' '.join(cleaned_string.split())
    return cleaned_string.strip()

def dice_similarity(str1, str2):
    str1 = clean_string(str1)
    str2 = clean_string(str2)
    set1 = set(str1.lower().split(' '))
    set2 = set(str2.lower().split(' '))
    intersection = len(set1 & set2)
    return (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0.0


def remove_duplicates_and_update_df(df: pd.DataFrame, labels: dict, threshold=0.75):
    updated_labels = {}
    replacement_map = {}
    for key, categories in labels.items():
        unique_categories = []
        category_map = {}
        for category in categories:
            found_similar = False
            for unique_category in unique_categories:
                if dice_similarity(category, unique_category) >= threshold:
                    found_similar = True
                    category_map[category] = unique_category
                    break
            if not found_similar:
                unique_categories.append(category)
                category_map[category] = category
        updated_labels[key] = set(unique_categories)
        replacement_map[key] = category_map
    for key, mapping in replacement_map.items():
        if key in df.columns:
            df[key] = df[key].replace(mapping)
    return df, updated_labels


def errors_nan(df: pd.DataFrame, colname: str):
    nan_rows = df[df[colname].isna()]  # Filter rows where the given column has NaN values
    result_dict = {}
    for row_index in nan_rows.index:
        col_index = df.columns.get_loc(colname)  # Get the index of the column dynamically
        result_dict[(row_index, col_index)] = 'N/A'
    return result_dict


def errors_zero(df: pd.DataFrame, colname: str):
    zero_rows = df[df[colname] == 0]  # Filter rows where the given column has 0 values
    result_dict = {}
    for row_index in zero_rows.index:
        col_index = df.columns.get_loc(colname)  # Get the index of the column dynamically
        result_dict[(row_index, col_index)] = 0  # Set the value to 0 where the condition is met
    return result_dict


def is_valid_date(value):
    try:
        pd.to_datetime(value, format='%Y-%m-%d', errors='raise')
        return True
    except (ValueError, TypeError):
        return False


def check_valid_dates_in_column(df: pd.DataFrame, colname: str):
    result_dict = {}
    invalid_rows = df[~df[colname].apply(is_valid_date)]  # Filter rows where the date is invalid
    for row_index in invalid_rows.index:
        col_index = df.columns.get_loc(colname)  # Get the column index
        result_dict[(row_index, col_index)] = invalid_rows[colname].loc[row_index]  # Store the actual invalid value
    return result_dict


def csv_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            e1, e2, e3 = row  # Extract the three columns
            result_dict[(e1, e2)] = e3
    return result_dict




def remove_special_characters(text):
    # Use regex to remove anything that's not a letter, number, or whitespace
    cleaned_text = re.sub(r'[^A-Za-z0-9\säöüßÄÖÜ]', ' ', text)
    return cleaned_text


def detect_errors_with_phunspell(text):
    """
    Detect spelling errors using phunspell (Hunspell) for German.
    """
    words = text.split()
    cleaned_text = remove_special_characters(text).split(" ")
    filtered_words = [word for word in cleaned_text if word.lower() != "covid"]
    misspelled = pspell.lookup_list(filtered_words)
    return len(misspelled) > 0


def detect_error_textual(df: pd.DataFrame, colname: str):
    errors_dict = {}
    col_idx = df.columns.get_loc(colname)
    for idx, value in df[colname].items():
        if isinstance(value, str):
            if detect_errors_with_phunspell(value):
                errors_dict[(idx, col_idx)] = value
        else:
            continue
    return errors_dict
