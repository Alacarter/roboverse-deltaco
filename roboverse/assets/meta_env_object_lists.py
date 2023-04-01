from collections import OrderedDict

# 16 pairs -- can make 32 tasks out of it
PICK_PLACE_TRAIN_TASK_OBJECTS = [
    ['conic_cup', 'fountain_vase'],
    ['circular_table', 'hex_deep_bowl'],
    ['smushed_dumbbell', 'square_prism_bin'],
    ['narrow_tray', 'colunnade_top'],
    ['stalagcite_chunk', 'bongo_drum_bowl'],
    ['pacifier_vase', 'beehive_funnel'],
    ['crooked_lid_trash_can', 'toilet_bowl'],
    ['pepsi_bottle', 'tongue_chair'],
    ['modern_canoe', 'pear_ringed_vase'],
    ['short_handle_cup', 'bullet_vase'],
    ['glass_half_gallon', 'flat_bottom_sack_vase'],
    ['trapezoidal_bin', 'vintage_canoe'],
    ['bathtub', 'flowery_half_donut'],
    ['t_cup', 'cookie_circular_lidless_tin'],
    ['box_sofa', 'two_layered_lampshade'],
    ['conic_bin', 'jar'],
    #   'aero_cylinder',
]

PICK_PLACE_TRAIN_TASK_OBJECTS_TO_SHAPE_MAP = {
    'conic_cup': 'cup',
    'fountain_vase': 'vase',
    'circular_table': 'chalice',
    'hex_deep_bowl': 'bowl',
    'smushed_dumbbell': 'chalice',
    'square_prism_bin': 'trapezoidal prism',
    'narrow_tray': 'trapezoidal prism',
    'colunnade_top': 'chalice',
    'stalagcite_chunk': 'freeform',
    'bongo_drum_bowl': 'bowl',
    'pacifier_vase': 'vase',
    'beehive_funnel': 'vase',
    'crooked_lid_trash_can': 'cylinder',
    'toilet_bowl': 'bowl',
    'pepsi_bottle': 'bottle',
    'tongue_chair': 'freeform',
    'modern_canoe': 'canoe',
    'pear_ringed_vase': 'vase',
    'short_handle_cup': 'cup',
    'bullet_vase': 'vase',
    'glass_half_gallon': 'bottle',
    'flat_bottom_sack_vase': 'vase',
    'trapezoidal_bin': 'trapezoidal prism',
    'vintage_canoe': 'canoe',
    'bathtub': 'bowl',
    'flowery_half_donut': 'round hole',
    't_cup': 'cup',
    'cookie_circular_lidless_tin': 'bowl',
    'box_sofa': 'trapezoidal prism',
    'two_layered_lampshade': 'round hole',
    'conic_bin': 'cup',
    'jar': 'cylinder',
}

PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP = {}
for obj, idf in PICK_PLACE_TRAIN_TASK_OBJECTS_TO_SHAPE_MAP.items():
    if idf not in PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP:
        PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP[idf] = []
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP[idf].append(obj)

# Reorder the keys
new_shape_key_ordering = [
    "vase", "chalice", "freeform", "bottle", "canoe", # train
    "cup", "bowl", "trapezoidal prism", "cylinder", "round hole"] # eval
assert (set(new_shape_key_ordering)
        == set(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP.keys()))
PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP_OLD = PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP
PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP = OrderedDict()
for key in new_shape_key_ordering:
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP[key] = list(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP_OLD[key])
# print(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP)

PICK_PLACE_TRAIN_TASK_OBJECTS_TO_COLOR_MAP = {
    'conic_cup': 'black and white',
    'fountain_vase': 'brown',
    'circular_table': 'blue',
    'hex_deep_bowl': 'brown',
    'smushed_dumbbell': 'black and white',
    'square_prism_bin': 'brown',
    'narrow_tray': 'brown',
    'colunnade_top': 'gray',
    'stalagcite_chunk': 'white',
    'bongo_drum_bowl': 'red',
    'pacifier_vase': 'white',
    'beehive_funnel': 'brown',
    'crooked_lid_trash_can': 'gray',
    'toilet_bowl': 'white',
    'pepsi_bottle': 'red',
    'tongue_chair': 'red',
    'modern_canoe': 'orange',
    'pear_ringed_vase': 'orange',
    'short_handle_cup': 'blue',
    'bullet_vase': 'brown',
    'glass_half_gallon': 'gray',
    'flat_bottom_sack_vase': 'brown',
    'trapezoidal_bin': 'brown',
    'vintage_canoe': 'brown', # ??
    'bathtub': 'gray',
    'flowery_half_donut': 'gray', # ??
    't_cup': 'blue',
    'cookie_circular_lidless_tin': 'brown',
    'box_sofa': 'brown',
    'two_layered_lampshade': 'yellow',
    'conic_bin': 'gray',
    'jar': 'blue',
}

PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP = {}
for obj, color in PICK_PLACE_TRAIN_TASK_OBJECTS_TO_COLOR_MAP.items():
    if color not in PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP:
        PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP[color] = []
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP[color].append(obj)
# print(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP)

COLOR_TO_RGBA_MAP = {
    # "black and white": ,
    "brown": (0.75, 0.63, 0.45, 1), # (0.85, 0.73, 0.55, 1),
    "blue": (0.47, 0.65, 0.83, 1),
    "gray": (0.73, 0.73, 0.73, 1),
    "white": (1, 1, 1, 1),
    "red": (0.8, 0.08, 0.08, 1),
    "orange": (0.89, 0.45, 0.13, 1),
    "yellow": (0.97, 0.82, 0.12, 1),
}

# object_name to color pair map
OBJECT_NAME_TO_COLOR_PAIR_MAP = {
    'conic_cup': ['brown', 'blue'], #'black and white',
    'fountain_vase': ['blue', 'gray'], # 'brown',
    'circular_table': ['brown', 'gray'], #'blue',
    'hex_deep_bowl': ['gray', 'white'], # 'brown',
    'smushed_dumbbell': ['brown', 'gray'], # 'black and white',
    'square_prism_bin': ['gray', 'white'], # 'brown',
    'narrow_tray': ['gray', 'white'], # 'brown',
    'colunnade_top': ['white', 'red'], # 'gray',
    'stalagcite_chunk': ['red', 'brown'], #'white',
    'bongo_drum_bowl': ['white', 'brown'], #'red',
    'pacifier_vase': ['brown', 'gray'], # 'white',
    'beehive_funnel': ['gray', 'white'], # 'brown',
    'crooked_lid_trash_can': ['white', 'red'], # 'gray',
    'toilet_bowl': ['red', 'orange'], #'white',
    'pepsi_bottle': ['orange', 'blue'], #'red',
    'tongue_chair': ['orange', 'blue'], # 'red',
    'modern_canoe': ['blue', 'brown'], # 'orange',
    'pear_ringed_vase': ['blue', 'brown'], # 'orange',
    'short_handle_cup': ['brown', 'gray'], # 'blue',
    'bullet_vase': ['gray', 'blue'], # 'brown',
    'glass_half_gallon': ['brown', 'blue'], # 'gray',
    'flat_bottom_sack_vase': ['gray', 'blue'], # 'brown',
    'trapezoidal_bin': ['gray', 'blue'], # 'brown',
    'vintage_canoe': ['gray', 'blue'], # 'brown', # ??
    'bathtub': ['blue', 'brown'], # 'gray',
    'flowery_half_donut': ['blue', 'brown'], # 'gray', # ??
    't_cup': ['brown', 'yellow'], # 'blue',
    'cookie_circular_lidless_tin': ['yellow', 'gray'], # 'brown',
    'box_sofa': ['yellow', 'gray'], # 'brown',
    'two_layered_lampshade': ['gray', 'blue'], # 'yellow',
    'conic_bin': ['blue', 'brown'], # 'gray',
    'jar': ['brown', 'gray'], # 'blue',
}
