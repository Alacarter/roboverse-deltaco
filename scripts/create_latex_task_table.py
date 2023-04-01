import pandas as pd
from roboverse.assets.meta_env_object_lists import (
    PICK_PLACE_TRAIN_TASK_OBJECTS,
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP,
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP,
)


def create_tasks_df(
        col_names, row_names, obj_idf_col_name, obj_idf_type_to_num_rows_map):
    col_to_task_idx_map = {}
    num_rows = len(row_names)

    col_to_task_idx_map[obj_idf_col_name] = []
    for obj_idf_type, num in obj_idf_type_to_num_rows_map.items():
        col_to_task_idx_map[obj_idf_col_name].extend([obj_idf_type] * num)

    col_to_task_idx_map['Object Identifier'] = row_names

    for i, col_name in enumerate(col_names):
        col_to_task_idx_map[col_name] = list(
            range(i * num_rows, (i + 1) * num_rows))
    df = pd.DataFrame(col_to_task_idx_map)
    # df = df.set_index('Object Identifier Type', append=True).swaplevel(0,1)
    return df


def add_hline(df_latex, obj_idf_type_to_num_rows_map, row_names):
    sep = "\\\\\n"
    df_latex_lines = df_latex.split(sep)
    # data_lines = df_latex_lines[1:-1]
    num_rows = len(row_names)
    assert len(df_latex_lines) == num_rows + 2

    # Create hline indices
    hline_indices = []
    prev_val = 0
    for val in obj_idf_type_to_num_rows_map.values():
        hline_indices.append(prev_val + val)
        prev_val += val

    # import ipdb; ipdb.set_trace()

    new_df_latex_lines = []
    # import ipdb; ipdb.set_trace()
    for i, line in enumerate(df_latex_lines):
        if i in hline_indices:
            line_maybe_with_hline = line + "\\\\ \hline \n"
        elif 1 <= i < 1 + num_rows:
            # A data row.
            line_maybe_with_hline = line + sep
        else:
            line_maybe_with_hline = line + "\n"
        new_df_latex_lines.append(line_maybe_with_hline)
    return "".join(new_df_latex_lines)


def change_col_spec(df_latex):
    new_col_spec = ("c|" * (len(col_names) + 2))[:-1]
    df_latex_lines = df_latex.split("\n")
    df_latex_lines[0] = "\\begin{tabular}" + "{" + new_col_spec + "}"
    df_latex = "\n".join(df_latex_lines)
    return df_latex


def add_line_after_column_header(df_latex, obj_idf_col_name):
    df_latex_lines = df_latex.split("\n")
    for i, line in enumerate(df_latex_lines):
        if obj_idf_col_name in line:
            df_latex_lines[i] = line + "\\\\ \hline"
    return "\n".join(df_latex_lines)


def get_iterable_item_from_line(line, lookup_iterable):
    for obj_idf in lookup_iterable:
        if obj_idf in line:
            return (line.index(obj_idf), line.index(obj_idf) + len(obj_idf), obj_idf)
    return (-1, -1, None)


def add_multirow(df_latex, obj_idf_type_to_num_rows_map):
    sep = "\\\\\n"
    df_latex_lines = df_latex.split(sep)
    new_df_latex_lines = []
    prev_obj_idf = None
    obj_idf_val_len = 0
    for i, line in enumerate(df_latex_lines):
        obj_idf_start, obj_idf_end, obj_idf = get_iterable_item_from_line(
            line, obj_idf_type_to_num_rows_map)
        if obj_idf != prev_obj_idf and obj_idf is not None:
            multirow_num_rows = obj_idf_type_to_num_rows_map[obj_idf]
            new_obj_idf_val = (
                "\\multirow{" + str(multirow_num_rows)
                + "}{*}{" + obj_idf + "}")
            obj_idf_val_len = len(new_obj_idf_val)
        else:
            new_obj_idf_val = " " * obj_idf_val_len
        modified_line = (
            line[:obj_idf_start]
            + new_obj_idf_val + line[obj_idf_end:])
        prev_obj_idf = obj_idf
        new_df_latex_lines.append(modified_line)
    return sep.join(new_df_latex_lines)


def add_rowcolor(df_latex, color_str_list, row_names):
    sep = "\\\\\n"
    df_latex_lines = df_latex.split(sep)
    new_df_latex_lines = []
    for i, line in enumerate(df_latex_lines):
        token_start, token_end, token = get_iterable_item_from_line(
            line, row_names)
        if i == 0:
            # Col names cannot be colored.
            new_line = line
        elif token is not None:
            # one of the 50 rows that must be colored.
            color_str = color_str_list[i-1]
            colored_cell_vals_line = " & ".join(
                ["\\cellcolor{" + color_str + "}" + cellval.strip()
                 for cellval in line[token_start:].split("&")])
            new_line = line[:token_start] + colored_cell_vals_line
        else:
            new_line = line
        new_df_latex_lines.append(new_line)
    return sep.join(new_df_latex_lines)


def replace_uscore_with_space(str_list):
    return [x.replace("_", " ") for x in str_list]


def flatten_list(lst):
    flattened_lst = []
    for elem in lst:
        if isinstance(flattened_lst, list):
            flattened_lst.extend(elem)
        else:
            flattened_lst.append(elem)
    return flattened_lst


def tokens_to_remove(df_latex, blacklist_tokens):
    for blacklist_token in blacklist_tokens:
        df_latex = df_latex.replace(blacklist_token, "")
    return df_latex


if __name__ == "__main__":
    # row_names = ["obja", "objb", "objc", "objd"]
    name_rows = flatten_list(PICK_PLACE_TRAIN_TASK_OBJECTS)
    color_rows = list(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP.keys())
    shape_rows = list(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP.keys())
    row_names = (
        name_rows + color_rows + shape_rows
    )
    col_names = ['green', 'red', 'front', 'back', 'left', 'right']
    color_str_list = (
        ["Gray"] * 24 + ["LightYellow"] * 8
        + ["LightBlue"] * 4 + ["Gray"] * 4
        + ["Gray"] * 5 + ["LightGreen"] * 5)
    obj_idf_col_name = 'Object Identifier Type'
    obj_idf_type_to_num_rows_map = {
        "name": len(name_rows),
        "color": len(color_rows),
        "shape": len(shape_rows)}

    row_names = replace_uscore_with_space(row_names)
    col_names = replace_uscore_with_space(col_names)
    assert (
        len(row_names) ==
        sum(obj_idf_type_to_num_rows_map.values()) ==
        len(color_str_list))
    df = create_tasks_df(
        col_names, row_names, obj_idf_col_name, obj_idf_type_to_num_rows_map)
    df_latex = df.to_latex(index=False, multirow=True)
    # df_latex = df_latex.replace("\\\\", "\\\\ \hline")
    df_latex = add_multirow(df_latex, obj_idf_type_to_num_rows_map)
    df_latex = add_rowcolor(df_latex, color_str_list, row_names)
    df_latex = add_hline(df_latex, obj_idf_type_to_num_rows_map, row_names)
    df_latex = change_col_spec(df_latex)
    df_latex = add_line_after_column_header(df_latex, obj_idf_col_name)
    df_latex = tokens_to_remove(
        df_latex, ["\\bottomrule\n", "\\midrule\n", "\\toprule\n"])
    print(df_latex)
