from PIL import Image
from io import BytesIO
import random
import os
import re
import operator

'''
Input directory: directory with stimuli to be transformed
Output directory: path of existing / new target directory for saving transformed stimuli
Next index: define starting index of new stimuli to standardize naming
Trials: trials to run, leave as None if max trials is preferred

Transformations:
"Counting": Apply a mathematical operation (addition, subtraction, multiplication, division)
    Parameters: 
        "+1","+2","+3",
        "-1","-2","-3",
        "x2","x3",
        "d2","d3"
        
"Resize": Resize horizontally(X), vertically(Y), or both ways(XY)
    Parameters: 
        "0.5X", "0.5Y", "0.5XY", 
        "2X", "2Y", "2XY"
        
"Colour": Colour Change
    Parameters: 
        "Red", "Yellow", "Green", "Blue", "Purple", "Grey"
        
"Reflect": Reflect along the X or Y axis
    Parameters: 
        "X", "Y"
        
"2DRotation": Rotate clockwise(+) by a certain degree
    Parameters: 
        45, 90, 135,180
'''
input_directory = "/Users/charliewong/Downloads/Objects16"
output_directory = "/Users/charliewong/Downloads/test"
next_index = 0
trials = 1
transformation = "Counting"
parameter = "+3"

# ------------------------------------------------------------------------------------------------

angles = [45,90,135,180]
factors = ["0.5X", "0.5Y", "0.5Y", "2X", "2Y", "2XY"]
operations = ["+1","+2","+3","-1","-2","-3","x2","x3","d2","d3"]
colours = ["Red","Yellow","Green","Blue","Purple","Grey"]
reflections = ["X","Y"]
add = sub = [1,2,3]
mul = div = [2,3]
train_1 = []
train_2 = []
test = []
transformations = ["Counting", "Resize", "Colour", "Reflect", "2DRotation"]
suffixes = (
    f"train_0_input.png",f"train_0_output.png",f"train_1_input.png",f"train_1_output.png",
    f"test_0_input.png",f"test_mc_0_input.png",f"test_mc_1_input.png",f"test_mc_2_input.png"
    )
    

out_directory = output_directory
if not os.path.isdir(out_directory):
    os.mkdir(out_directory)

in_directory = input_directory
entries = os.listdir(in_directory)
paths = [os.path.join(in_directory, entry) for entry in entries]
filtered_paths = [path for path in paths if not path.endswith(".DS_Store")]
remainder = len(filtered_paths) % 3
if remainder != 0:
    filtered_paths = filtered_paths[:-(remainder)]
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

def save_txt(train_0_in,train_0_out,train_1_in,train_1_out,test_in,mc1,mc2):
    with open(f"output_{transformation}{parameter}.txt", "a") as file:
        file.write(f'train_0_input: {train_0_in}\n')
        file.write(f'train_0_output: {train_0_out}\n')
        file.write(f'train_1_input: {train_1_in}\n')
        file.write(f'train_1_output: {train_1_out}\n')
        file.write(f'test_input: {test_in}\n')
        file.write(f'mc_1: {mc1}\n')
        file.write(f'mc_2: {mc2}\n')

def crop(image_input):
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    
    new_size = min(img.size)
    left = (img.width - new_size) / 2
    top = (img.height - new_size) / 2
    right = (img.width + new_size) / 2
    bottom = (img.height + new_size) / 2
    cropped = img.crop((left, top, right, bottom))
    
    return cropped

def save_image(image_input, transformation, param, index, file_suffix):
    for image, suffix in zip(image_input,file_suffix):
        cropped = crop(image)
        file_path = os.path.join(out_directory, f"{transformation}{param}_{index}_{suffix}")
        cropped.save(file_path,format="PNG")

def selector(to_exclude, parameters, num):
    filtered = [parameter for parameter in parameters if parameter != to_exclude]
    selected = random.sample(filtered,num)
    return selected  

def reflect_image(img_path, axis):
    with Image.open(img_path) as image:
        if axis == 'Y':
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif axis == 'X':
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("Invalid Parameter")
        return flipped_image

def rotate_image(img_path, angle):
    if not isinstance(angle,int):
        raise ValueError("Invalid Parameter")
    with Image.open(img_path) as image:
        rotated_image = image.rotate(-angle, expand=False)
        return rotated_image

def resize(img_path, factor, original = False):
    if factor not in factors:
        raise ValueError("Invalid Parameter")
    
    resize_factor = re.findall(r'[\d.]+', factor)[0]
    axis = re.findall(r'[A-Za-z]+', factor)[0]
    if "." in resize_factor:
        resize_factor = 0.5 
    else:
        resize_factor = 2  
    
    initial_shrink_factor = 0.5
    
    factor_x = factor_y = initial_shrink_factor  
    if not original:
        if axis == "X":
            factor_x *= resize_factor
        elif axis == "Y":
            factor_y *= resize_factor 
        elif axis == "XY": 
            factor_x *= resize_factor
            factor_y *= resize_factor
    
    with Image.open(img_path) as img:
        new_size = (int(img.width * factor_x), int(img.height * factor_y))
        resized_img = img.resize(new_size, Image.LANCZOS)
        
        if img.mode == 'RGBA' or (img.mode == 'P' and 'transparency' in img.info):
            # Create a transparent background instead of white
            background = Image.new('RGBA', (img.width, img.height), (255, 255, 255, 0))
        else:
            # For non-transparent images, create a white background
            background = Image.new('RGB', (img.width, img.height), (255, 255, 255))
        
        upper_left_x = (background.width - resized_img.width) // 2
        upper_left_y = (background.height - resized_img.height) // 2
        
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background.paste(resized_img, (upper_left_x, upper_left_y), resized_img)
        else:
            background.paste(resized_img, (upper_left_x, upper_left_y))
        
        return background


def colour_change(img_path, colour_name):
    colour_map = {
        'Red': (255, 0, 0, 200),  
        'Yellow': (255, 255, 0, 200),
        'Green': (0, 128, 0, 200), 
        'Blue': (0, 0, 255, 200),
        'Purple': (128, 0,128,200),
    }
    
    #fix thiscondition, raise valueerror if its anything but whats in colours
    if colour_name != "Grey":
        colour = colour_map.get(colour_name) 
    else:
        colour = "Grey"           
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
                    if colour != "Grey":
                        new_colour = (
                            int((original_pixel[0] + colour[0]) / 2),
                            int((original_pixel[1] + colour[1]) / 2),
                            int((original_pixel[2] + colour[2]) / 2),
                            original_pixel[3],
                        )
                    else:
                        luminance = int(original_pixel[0] * 0.299 + original_pixel[1] * 0.587 + original_pixel[2] * 0.114)
                        new_colour = (
                            luminance,
                            luminance,
                            luminance,
                            original_pixel[3],
                        )
                    pixels[x, y] = new_colour
                else:
                    pixels[x, y] = original_pixel
    return result_img

def count_builder(oper):
    if oper not in operations:
        raise ValueError("Invalid Parameter")
    num = int(oper[1:])
    math_operations = {
        '+': operator.add,
        '-': operator.sub,
        'x': operator.mul,
        'd': operator.truediv
    }
    if oper.startswith('x'):
        starting = [1,2,3]
        math_op_name = 'x'
        math_op = math_operations['x']
    elif oper.startswith('+'):
        starting = [1,2,3,4,5]
        math_op_name = '+'
        math_op = math_operations['+']
    elif oper.startswith('-'):
        starting = [9,8,7,6,5]
        math_op_name = '-'
        math_op = math_operations['-']
    else:
        math_op_name = 'd'
        math_op = math_operations['d']
        if num%2 == 0:
            starting = [6,4,2]
        else:
            starting = [9,6,3]
    mc1_op = math_operations["+"]
    mc2_op = math_operations["-"]
    train1, train2, test_0 = random.sample(starting,3)
    train1_out = math_op(train1, num)
    train2_out = math_op(train2, num)
    test_out = math_op(test_0, num)
    print(test_out)
    test_mc1_out = mc1_op(test_out,1)
    print(test_mc1_out)
    test_mc2_out = mc2_op(test_out,1)
    print(test_mc2_out)
    save_txt(train1,train1_out,train2,train2_out,
        test_0,"+1","-1")
    return (train1,train1_out,train2,train2_out,test_0,test_out,test_mc1_out,test_mc2_out) 
    
def count_generator(img_path, num):
    num = int(num)
    img = crop(img_path)
    canvas = Image.new('RGBA', (img.width, img.height), (0, 0, 0, 0))
    max_items_per_row = int((10 ** 0.5) + 1)
    item_size = min(img.width, img.height) // max_items_per_row
    
    for i in range(num):
        x = (i % max_items_per_row) * item_size
        y = (i // max_items_per_row) * item_size
        shrunken_img = img.resize((item_size, item_size), Image.LANCZOS)
        mask = None
        canvas.paste(shrunken_img, (x, y), mask)
    
    return canvas

def transform_save_count(index, param, next_index=0):
    results = count_builder(param)
    images = (
        train_1[index],train_1[index],train_2[index],train_2[index],
        test[index],test[index],test[index],test[index]
    )
    inputs = [None]*8
    for i in range(8):
        inputs[i] = count_generator(images[i],results[i])
    save_image(inputs, "Counting", param, index + next_index, suffixes)
    
def transform_save_colour(index, param, inputs, next_index):
    processed_1_in = colour_change(train_1[index], inputs[0])
    processed_1_out = colour_change(train_1[index], param)
    processed_2_in = colour_change(train_2[index],inputs[1])
    processed_2_out = colour_change(train_2[index], param)
    processed_test = colour_change(test[index], inputs[2])
    processed_mc_0 = colour_change(test[index], param)
    processed_mc_1 = colour_change(test[index], inputs[3])
    processed_mc_2 = colour_change(test[index], inputs[4])
    save_inputs = (
        processed_1_in, processed_1_out, processed_2_in, processed_2_out,
        processed_test, processed_mc_0, processed_mc_1, processed_mc_2
        )
    save_image(save_inputs,"Colour", param, index+next_index, suffixes)
    save_txt(inputs[0],param,inputs[1],param,inputs[2],inputs[3],inputs[4])
    
def transform_save_resize(index, param, inputs, next_index):
    processed_1_in = resize(train_1[index], param, original=True)
    processed_1 = resize(train_1[index], param)
    processed_2_in = resize(train_2[index], param, original=True)
    processed_2 = resize(train_2[index], param)
    processed_test = resize(test[index], param, original=True)
    processed_mc_0 = resize(test[index], param)
    processed_mc_1 = resize(test[index], inputs[0])
    processed_mc_2 = resize(test[index], inputs[1])
    save_inputs = (
        processed_1_in, processed_1, processed_2_in, processed_2, 
        processed_test, processed_mc_0, processed_mc_1, processed_mc_2
        )
    save_image(save_inputs, "Resize", param, index+next_index, suffixes)
    save_txt(" ",param," ",param," ",inputs[0],inputs[1])

def transform_save_reflect(index, param, false_axis, angle, next_index):
    processed_1 = reflect_image(train_1[index], param)
    processed_2 = reflect_image(train_2[index], param)
    processed_mc_0 = reflect_image(test[index], param)
    processed_mc_1 = reflect_image(test[index], false_axis)
    processed_mc_2 = rotate_image(test[index], angle)
    save_inputs = (
        train_1[index], processed_1, train_2[index], processed_2, 
        test[index], processed_mc_0, processed_mc_1, processed_mc_2
        )
    save_image(save_inputs, "Reflect", param, index+next_index, suffixes)
    save_txt("0",param,"0",param,"0",false_axis,angle)

def transform_save_rotate(index, param, inputs, next_index):
    processed_1_in = rotate_image(train_1[index], inputs[0])
    processed_1_out = rotate_image(train_1[index], inputs[0]+param)
    processed_2_in = rotate_image(train_2[index],inputs[1])
    processed_2_out = rotate_image(train_2[index], inputs[1]+param)
    processed_test = rotate_image(test[index], inputs[2])
    processed_mc_0 = rotate_image(test[index], inputs[2]+param)
    processed_mc_1 = rotate_image(test[index], inputs[2]+inputs[3])
    processed_mc_2 = rotate_image(test[index], inputs[2]+inputs[4])
    save_inputs = (
        processed_1_in, processed_1_out, processed_2_in, processed_2_out,
        processed_test, processed_mc_0, processed_mc_1, processed_mc_2
        )
    save_image(save_inputs,"2DRotation", param, index+next_index, suffixes)
    save_txt(inputs[0],inputs[0]+param,inputs[1],inputs[1]+param,inputs[2],inputs[3],inputs[4])

if transformation not in transformation:
    raise ValueError(f"Transformation not found")

elif transformation == "Reflect":
    axis = parameter
    false_axis = "Y" if axis == "X" else "X"
    reflect_angles = [45,90,135]
    angle = random.choice(reflect_angles)
    for i in range(sublist_len):
        angle = random.choice(reflect_angles)
        transform_save_reflect(i, axis, false_axis, angle, next_index)
        
elif transformation == "Colour":
    colour = parameter 
    for i in range(sublist_len):
        train1, train2, test_0, colour_mc_1,colour_mc_2 = selector(colour,colours,5)
        input_params = (train1,train2,test_0,colour_mc_1,colour_mc_2)
        transform_save_colour(i, colour, input_params, next_index)
        
elif transformation == "Resize":
    factor = parameter
    for i in range(sublist_len):
        factor_mc_1,factor_mc_2 = selector(factor,factors, 2)
        input_params = (factor_mc_1,factor_mc_2)
        transform_save_resize(i, factor, input_params, next_index)
            
elif transformation == "2DRotation":
    angle = parameter
    for i in range(sublist_len):
        angle_mc_1, angle_mc_2 = selector(angle, angles[:-1], 2)
        train1, train2, test_0 = random.sample(angles, 3)
        input_params = (train1,train2,test_0,angle_mc_1,angle_mc_2)
        transform_save_rotate(i, angle, input_params, next_index)
        
else:
    operation = parameter
    for i in range(sublist_len):
        transform_save_count(i, operation, next_index)
    