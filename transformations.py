from PIL import Image
from io import BytesIO
import random
import os
import numpy as np

'''
Input directory: directory with stimuli to be transformed
Output directory: path of existing / new target directory for saving transformed stimuli
Next index: define starting index of new stimuli to standardize naming
Trials: trials to run, leave as None if max trials is preferred

Transformations:
"Counting"
    Parameters: "+1","+2","+3","x3","-1","-2","-3","/2","/3"
"Resize"
    Parameters: 1/4, 1/3, 1/2, 3/4, 2, 3, 4, 5
"Colour"
    Parameters: "Red", "Yellow", "Green", "Blue"
"Reflect"
    Parameters: "X", "Y"
"2DRotation"
    Parameters: 30, 60, 90, 120, 210, 240, 270, 300, 330, 360
'''

input_directory = "/Users/charliewong/Downloads/RRRObjects"
output_directory = "/Users/charliewong/Downloads/test"
next_index = 0
trials = None
transformation = " "
parameter = None

# ------------------------------------------------------------------------------------------------

angles = [30, 60, 90, 120, 210, 240, 270, 300, 330, 360]
factors = [1/4,1/3,1/2,3/4,2,3,4,5]
operations_1 = ["+1","+2","+3","x3"]
operations_2 = ["-1","-2","-3","/2","/3"]
colours = ["Red","Yellow","Green","Blue"]
reflections = ["X","Y"]
train_1 = []
train_2 = []
test = []
transformations = ["Counting", "Resize", "Colour", "Reflect", "2DRotation"]

out_directory = output_directory
if not os.path.isdir(out_directory):
    os.mkdir(out_directory)

in_directory = input_directory
entries = os.listdir(in_directory)
paths = [os.path.join(in_directory, entry) for entry in entries]
filtered_paths = [path for path in paths if not path.endswith(".DS_Store")]
filtered_paths = filtered_paths[:-(len(filtered_paths)%3)]
sublist_len = int(len(filtered_paths)/3)

def create_sets(trials):
    train_1 = filtered_paths[:trials]
    train_2 = filtered_paths[trials : (2*trials)]
    test = filtered_paths[(2*trials) : (3*trials)]
    return train_1,train_2, test

if trials is None:
    trials = sublist_len
    train_1,train_2,test = create_sets(trials)
elif trials <= sublist_len and trials > 0:
    sublist_len = trials
    train_1,train_2,test = create_sets(trials)
else:
    raise ValueError(f"The maximum number of trials possible is {sublist_len}.")
    
def save_image(image_path, transformation, param, index, file_suffix):
    with Image.open(image_path) as image:
        file_path = os.path.join(out_directory, f"{transformation}{param}_{index}_{file_suffix}")
        image.save(file_path)

def save_processed_image(image, transformation, param, index, file_suffix):
    file_path = os.path.join(out_directory, f"{transformation}{param}_{index}_{file_suffix}")
    image.save(file_path)

def reflect_image(img_path, axis):
    with Image.open(img_path) as image:
        if axis == 'Y':
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif axis == 'X':
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("Axis must be 'X' or 'Y'")
        return flipped_image

def rotate_image(img_path, angle):
    with Image.open(img_path) as image:
        rotated_image = image.rotate(angle, expand=True)
        return rotated_image

def resize(img_path, factor):
    with Image.open(img_path) as img:
        new_size = (int(img.width * factor), int(img.height * factor))
        resized_img = img.resize(new_size, Image.LANCZOS)
        
        background = Image.new('RGB', (img.width, img.height), (255, 255, 255))

        upper_left_x = (background.width - resized_img.width) // 2
        upper_left_y = (background.height - resized_img.height) // 2

        background.paste(resized_img, (upper_left_x, upper_left_y))

        return background

def colour_change(img_path, colour_name):
    colour_map = {
        'red': (255, 0, 0, 200),  
        'yellow': (255, 255, 0, 200),
        'green': (0, 128, 0, 200), 
        'blue': (0, 0, 255, 200)  
    }

    colour = colour_map.get(colour_name.lower())
    if colour is None:
        raise ValueError(f"Color '{colour_name}' is not one of the defined colors.")
    
    with Image.open(img_path).convert("RGBA") as img:
        width, height = img.size

        result_img = Image.new('RGBA', img.size)
        pixels = result_img.load()

        for x in range(width):
            for y in range(height):
                original_pixel = img.getpixel((x, y))
                
                if original_pixel[3] > 0: 
                    new_color = (
                        int((original_pixel[0] + colour[0]) / 2),
                        int((original_pixel[1] + colour[1]) / 2),
                        int((original_pixel[2] + colour[2]) / 2),
                        original_pixel[3],
                    )
                    pixels[x, y] = new_color
                else:
                    pixels[x, y] = original_pixel

    return result_img
    
def counting(img_path, operation, objects_per_row=3):
    with Image.open(img_path) as img:
        if operation.startswith('x'):
            num_objects = int(operation[1:])
        elif operation.startswith('+'):
            num_objects = int(operation[1:]) + 1
        else:
            num_objects = 1 

        num_rows = (num_objects + objects_per_row - 1) // objects_per_row
        num_columns = min(num_objects, objects_per_row)
        
        new_width = img.width * num_columns
        new_height = img.height * num_rows
        
        background = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        
        for i in range(num_objects):
            row = i // objects_per_row
            column = i % objects_per_row
            offset_x = img.width * column
            offset_y = img.height * row
            background.paste(img, (offset_x, offset_y))
        
        return background

def select_colour(exclude_colour, colours=colours):
    filtered_colours = [color_name for color_name in colours if color_name != exclude_colour]
    selected_colours = random.sample(filtered_colours, 3)
    return selected_colours[0], selected_colours[1], selected_colours[2]

def modify_operation(operation):
    if operation.startswith('-'):
        return '+' + operation[1:]
    elif operation.startswith('/'):
        return 'x' + operation[1:]
    else:
        return operation

def select(x,list):
    filtered_numbers = [a for a in list if a != x]
    selection = random.sample(filtered_numbers, 2)
    return selection[0],selection[1]

def transform_save(index, transformation, param, process_image, mc_1, mc_2, next_index=0, input_colour=None, mod=False, reflect=False):
    train_sets = [train_1, train_2]
    original_suffix = "output.png" if mod else "input.png"
    processed_suffix = "input.png" if mod else "output.png"

    for i, train_set in enumerate(train_sets, start=1):
        image_path = train_set[index]
        processing_param = input_colour if input_colour is not None else param
        
        processed_in = process_image(image_path, processing_param)
        processed_out = process_image(image_path, param) if input_colour is not None else processed_in
        
        save_processed_image(processed_in, transformation, processing_param, index + next_index, f"train_{i}_{original_suffix}")
        save_processed_image(processed_out, transformation, param, index + next_index, f"train_{i}_{processed_suffix}")
        
        if input_colour is None:
            save_image(image_path, transformation, param, index + next_index, f"train_{i}_{original_suffix}")

    test_image_path = test[index]
    if input_colour is None:
        save_image(test_image_path, transformation, param, index + next_index, f"test_0_{original_suffix}")
    else:
        processed_test = process_image(test_image_path, input_colour)
        save_processed_image(processed_test, transformation, param, index + next_index, f"test_0_{original_suffix}")

    for mc_index, mc_param in enumerate([param, mc_1, mc_2], start=0):
        if reflect and mc_param == mc_2:
            processed = rotate_image(test_image_path, mc_2)
        else:
            processed = process_image(test_image_path, mc_param)
        suffix = "input.png" if mod and mc_index == 0 else f"mc_{mc_index}_output.png" if mod else f"mc_{mc_index}_input.png"
        save_processed_image(processed, transformation, param, index + next_index, f"test_{suffix}")

if transformation not in transformation:
    raise ValueError(f"Transformation not found")

if transformation == "Reflect":
    axis = parameter
    false_axis = "Y" if axis == "X" else "X"
    angle = random.choice(angles[:-1])
    for i in range(sublist_len):
        transform_save(i, "Reflect", axis, reflect_image, false_axis, angle, next_index, reflect = True)
elif transformation == "Colour":
    colour = parameter
    colour_mc_1,colour_mc_2, in_colour = select_colour(colour,colours)
    for i in range(sublist_len):
        transform_save(i, "Colour", colour, colour_change, colour_mc_1, colour_mc_2, next_index, in_colour)
elif transformation == "Resize":
    factor = parameter
    for i in range(sublist_len):
        if factor > 1:
            factor_mc_1,factor_mc_2 = select(np.round(1/factor, decimals = 2),factors[:4])
            transform_save(i, "Resize", np.round(1/factor, decimals = 2), resize, factor_mc_1, factor_mc_2, next_index, mod=True)
        else:
            factor_mc_1,factor_mc_2 = select(factor,factors[:4])
            transform_save(i, "Resize", factor, resize, factor_mc_1, factor_mc_2, next_index)
elif transformation == "2DRotation":
    angle = parameter
    angle_mc_1, angle_mc_2 = select(angle,angles)
    for i in range(sublist_len):
        transform_save(i, "2DRotation", angle, rotate_image, angle_mc_1, angle_mc_2, next_index)
else:
    operation_mc_1 = "+5"
    operation_mc_2 = "x4"
    operation = parameter
    for i in range(sublist_len):
        if operation.startswith('-') or operation.startswith('/'):
            new_oper = modify_operation(operation)
            selection = random.sample(operations_1, 2)
            transform_save(i, "Counting", new_oper, counting, operation_mc_1, operation_mc_2, next_index, mod=True)
        else:
            transform_save(i, "Counting", operation, counting, operation_mc_1, operation_mc_2, next_index)
    