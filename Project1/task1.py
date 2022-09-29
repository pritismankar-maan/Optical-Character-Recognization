"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
from enum import unique
from itertools import count
import json
import os
import glob
import cv2
from cv2 import COLOR_BGR2GRAY
from cv2 import resize
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    
    # thresholding
    height, width = test_img.shape
    for i in range(height):
        for j in range(width):
            if test_img[i,j] < 191:
                test_img[i,j] = 0   # characters
            else:
                test_img[i,j] = 255  # background
    
    # feature detection
    features_array = enrollment(characters,test_img)

    total_labels,bounding_box_data =detection(test_img,height,width)
    # total_labels = 143
    # bounding_box_data = []

    results = recognition(total_labels,bounding_box_data,features_array)
    
    return results
    #raise NotImplementedError

def enrollment(characters,test_img):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    total_descriptor = []
    features_array = []
    
    for character in characters:
        # thresholding
        height, width = character[1].shape
        for i in range(height):
            for j in range(width):
                if character[1][i,j] < 190:
                    character[1][i,j] = 0   # characters
                else:
                    character[1][i,j] = 255  # background
        
        # create bounding box
        height_box = 0
        width_box = 0
        got_start_y = False
        got_start_x = False
        for i in range(height):
            if (character[1][i,:] == 0).any():
                if got_start_y == False:
                    start_y = i
                    got_start_y = True
            else:    
                if got_start_y == True:
                    height_box = i-start_y
                    break
        
        for j in range(width):
            if (character[1][:,j] == 0).any():
                if got_start_x == False:
                    start_x = j
                    got_start_x = True
            else:
                if got_start_x == True:
                    width_box = j - start_x    
                    break            
        
        # bounding box image
        box_char_img = character[1][start_y:(start_y+height_box),start_x:(start_x+width_box)]
        # extract features
        black_pixel_in_box = zoning(box_char_img)
        
        black_pixel_in_box_4half =  zoning_into4parts(box_char_img)

        size_ratio = height_box/width_box
        features_array.append([character[0],black_pixel_in_box,size_ratio,black_pixel_in_box_4half])
        

    return features_array
    
    #raise NotImplementedError

def zoning(input_image):
    size = 20
    zone_size = 4
    block = []
    resize_image = cv2.resize(input_image,(size,size))
    # thresholding
    for i in range(size):
        for j in range(size):
            if resize_image[i,j] < 190:
                resize_image[i,j] = 0
            else:
                resize_image[i,j] = 255
    
    block_array = []
    count = 0
    # divide into zones
    for i in range(size):
        for j in range(size):
            if i%zone_size==0 and j%zone_size==0:
                block = resize_image[i:i+zone_size,j:j+zone_size]
                block_array.append(block)
                count +=1
    
    # get the number of black pixel in all these zones
    black_pixels_in_box = []
    for block in block_array:
        for i in range(zone_size):
            for j in range(zone_size):
                if block[i,j] == 0:
                    count +=1
        
        black_pixels_in_box.append(count)
    
    return black_pixels_in_box

def zoning_into4parts(input_image):
    size = 20
    zone_size = 10
    block = []
    resize_image = cv2.resize(input_image,(size,size))
    # thresholding
    for i in range(size):
        for j in range(size):
            if resize_image[i,j] < 190:
                resize_image[i,j] = 0
            else:
                resize_image[i,j] = 255
    
    block_array = []
    count = 0
    # divide into zones
    for i in range(size):
        for j in range(size):
            if i%zone_size==0 and j%zone_size==0:
                block = resize_image[i:i+zone_size,j:j+zone_size]
                block_array.append(block)
                count +=1
    
    # get the number of black pixel in all these zones
    black_pixels_in_box = []
    for block in block_array:
        for i in range(zone_size):
            for j in range(zone_size):
                if block[i,j] == 0:
                    count +=1
        
        black_pixels_in_box.append(count)
    
    return black_pixels_in_box

def detection(test_img,height,width):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
        
    label_image = np.zeros([height,width])
    label_count = 0
    
    child_parent = []

    # first pass
    for i in range(1,height-1):
        for j in range(1,width-1):
            if test_img[i,j] == 0:  ## denote dark pixel which corresponds to character
                # if both top pixel and left pixel doesn't have a label
                if label_image[i-1,j] == 0 and label_image[i,j-1] ==0:
                    label_count += 1
                    label_image[i,j] = label_count
                
                else:
                    left = label_image[i-1,j]
                    top = label_image[i,j-1]
                    
                    if top == left and top != 0:
                        label_image[i,j] = top
                    else:
                        values = np.array([left, top])
                        label_image[i,j] = np.min(values[np.nonzero(values)])

                        if label_image[i,j] != left and left != 0:
                            # dont append if data is already present in child-parent table
                            is_present = False
                            for row in range(len(child_parent)-1):
                                if left in child_parent[row] and label_image[i,j] in child_parent[row]:
                                    is_present = True
                                    break
                            
                            if is_present == False:
                                child_list = []
                                parent_list = []
                                max_list = []
                                child_list_iter = 0
                                parent_list_iter = 0
                                row_number = 0
                                max_list_iter = 0
                                for list in child_parent:
                                    for list_list in list:
                                        if list_list == left:
                                            child_list = list
                                            child_list_iter = row_number
                                        if list_list == label_image[i,j]:
                                            parent_list = list
                                            parent_list_iter = row_number
                                        if list_list == top:
                                            max_list = list
                                            max_list_iter = row_number    
                                    row_number += 1    
                                
                                if len(child_list) == 0 and len(parent_list) == 0:
                                    child_parent.append([left,label_image[i,j]])
                                elif len(child_list) !=0  and len(parent_list) == 0:
                                    child_parent[child_list_iter].append(label_image[i,j])
                                elif len(child_list) ==0  and len(parent_list) != 0:
                                    child_parent[parent_list_iter].append(left)
                                else:
                                    if len(child_list) == len(parent_list):
                                        if child_list != parent_list:
                                            child_parent[parent_list_iter] = parent_list + child_list
                                    else:
                                        child_parent[parent_list_iter] = parent_list + child_list                                          
                                
                                if len(child_list) != 0 and len(parent_list) != 0 and top != 0:
                                    child_parent[parent_list_iter].append(min(max_list))
                                if len(parent_list) == 0 and len(max_list) == 0 and top != 0 and label_image[i,j] != 0:
                                    child_parent.append([top,label_image[i,j]])
                                                                
                        if label_image[i,j] != top and top != 0:
                            # dont append if data is already present in child-parent table
                            is_present = False
                            for row in range(len(child_parent)-1):
                                if top in child_parent[row] and label_image[i,j] in child_parent[row]:
                                    is_present = True
                                    break

                            if is_present == False:
                                child_list = []
                                parent_list = []
                                child_list_iter = 0
                                parent_list_iter = 0
                                max_list_iter = 0
                                max_list = []
                                row_number = 0
                                for list in child_parent:
                                    for list_list in list:
                                        if list_list == top:
                                            child_list = list
                                            child_list_iter = row_number
                                        if list_list == label_image[i,j]:
                                            parent_list = list
                                            parent_list_iter = row_number
                                        if list_list == left:
                                            max_list = list
                                            max_list_iter = row_number
                                    row_number += 1    
                                
                                if len(child_list) == 0 and len(parent_list) == 0:
                                    child_parent.append([top,label_image[i,j]])
                                elif len(child_list) !=0  and len(parent_list) == 0:
                                    child_parent[child_list_iter].append(label_image[i,j])
                                elif len(child_list) ==0  and len(parent_list) != 0:
                                    child_parent[parent_list_iter].append(top)
                                else:
                                    if len(child_list) == len(parent_list):
                                        if child_list != parent_list:
                                            child_parent[parent_list_iter] = parent_list + child_list
                                    else:
                                        child_parent[parent_list_iter] = parent_list + child_list
                                
                                if len(child_list) != 0 and len(parent_list) != 0 and left != 0:
                                    child_parent[parent_list_iter].append(min(max_list))
                                if len(parent_list) == 0 and len(max_list) == 0 and left != 0 and label_image[i,j] != 0:
                                    child_parent.append([left,label_image[i,j]])    
                            
    # check if list inside lists are disjoint or not
    for row in range(len(child_parent)-1):
        for list_list in child_parent[row]:
            for row2 in range(row+1,len(child_parent)-1):
                if list_list in child_parent[row2]:
                    child_parent[row] = child_parent[row] + child_parent[row2]    

    # add root parent for each row at the end
    for row in range(len(child_parent)-1):
        child_parent[row].append(min(child_parent[row]))

    # second pass - check if the current label is present in multiple rows of updated child_parent
    # assign the minimum of those root parents as the current label
    for i in range(1,height-1):
        for j in range(1,width-1):
            if label_image[i,j] != 0:

                curr_parent_array = []
                length = 0

                curr_parent = label_image[i,j]
                for row in range(len(child_parent)-1):
                    if curr_parent in child_parent[row]:
                        length = len(child_parent[row])
                        curr_parent = child_parent[row][length-1]
                        curr_parent_array.append(curr_parent)
                        
                
                # this child is the root after break statement     
                if len(curr_parent_array) == 0:
                    label_image[i,j] = curr_parent
                else:
                    label_image[i,j] = min(curr_parent_array)    


    # # print unique values and its shape including zero which is not a label
    # print(np.unique(label_image))
    # print(np.unique(label_image).shape)

    # get unique labels
    unique_labels = np.unique(label_image)
    total_labels = np.unique(label_image).shape[0]

    bounding_box_data = []
    for curr_label in range(1,total_labels):
        
        height_box = 0
        width_box = 0
        got_start_y = False
        got_start_x = False
        for i in range(height-1):
            if (label_image[i,:] == unique_labels[curr_label]).any():
                if got_start_y == False:
                    start_y = i
                    got_start_y = True
            else:    
                if got_start_y == True:
                    height_box = i-start_y
                    break
        
        for j in range(width-1):
            if (label_image[:,j] == unique_labels[curr_label]).any():
                if got_start_x == False:
                    start_x = j
                    got_start_x = True
            else:
                if got_start_x == True:
                    width_box = j - start_x    
                    break            
        
        bounding_box_data.append([start_x, start_y, width_box, height_box])

        box_image = test_img[start_y:(start_y+height_box),start_x:(start_x+width_box)]
        filename1 = './features/' + str(curr_label) + '_temp_file' + '.jpg'
        cv2.imwrite(filename1,box_image)

    return total_labels,bounding_box_data                   
    #raise NotImplementedError

def recognition(total_labels,bounding_box_data,features_array):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    
    recognised_char = []
    for i in range(1,total_labels):
        # extract each character from test image
        filename1 = './features/' + str(i) + '_temp_file' + '.jpg'
        curr_box_image = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
        
        black_pixels_in_box = zoning(curr_box_image)

        black_pixels_in_box_4half = zoning_into4parts(curr_box_image)

        sum_square_diff = 0
        ssd_array = []
        sum_square_4half_diff = 0
        ssd_4half_array = []
        # compare box filter zones with features of different characters
        for feature in features_array:
            # zoning with 25 zones
            black_pixels_in_box_array = np.array(black_pixels_in_box)
            feature_array = np.array(feature[1])

            diff_in_black_pixel_array = np.subtract(black_pixels_in_box_array,feature_array)
            
            square_diff = np.multiply(diff_in_black_pixel_array,diff_in_black_pixel_array)
            sum_square_diff = np.sum(square_diff)

            ssd_array.append(sum_square_diff)

            # zoning with 4 zones
            black_pixels_in_box_4half_array = np.array(black_pixels_in_box_4half)
            feature_4half_array = np.array(feature[3])

            diff_in_black_pixel_4half_array = np.subtract(black_pixels_in_box_4half_array,feature_4half_array)
            
            square_4half_diff = np.multiply(diff_in_black_pixel_4half_array,diff_in_black_pixel_4half_array)
            sum_square_4half_diff = np.sum(square_4half_diff)

            ssd_4half_array.append(sum_square_4half_diff)

                    
        ssd_min = min(ssd_array)
        min_index = ssd_array.index(ssd_min)
        if ssd_min < 1500:                                                              # 1st threshold
            height,width = curr_box_image.shape
            curr_size_ratio = height/width
            final_ratio = curr_size_ratio/features_array[min_index][2]
            if final_ratio > 0.5 and final_ratio <2:                                    # 2nd threshold
                ssd_4half_value = ssd_4half_array[min_index]
                if ssd_4half_value < 10000000:                                              # 3rd threshold
                    recognised_char.append(features_array[min_index][0])
                else:
                    recognised_char.append("UNKNOWN")        
            else:
                recognised_char.append("UNKNOWN")        
        else:
            recognised_char.append("UNKNOWN")    
             
    results = []
    for i in range(1,total_labels):
        my_dict = {"bbox": [bounding_box_data[i-1][0],bounding_box_data[i-1][1],bounding_box_data[i-1][2],bounding_box_data[i-1][3]],"name": recognised_char[i-1]}
        results.append(my_dict)
    
    
    return results
    #raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
