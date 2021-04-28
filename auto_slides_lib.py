
import re
import pandas as pd
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import Image, ImageOps
from resizeimage import resizeimage

from os import listdir
from os.path import isfile, join
import os 

import numpy as np
import cv2

import eng_to_ipa as ipa

from sys import exit


#todo: kiem tra khoang trang, case sensitive
class Excel:
    # Contructor
    def __init__(self, path, sheet_names):
        self.sheets = self.load_dtb(path, sheet_names) # a dict to contain all sheets data frame
    # load dtb from excel or pickle files
    @staticmethod
    def load_dtb(path, sheet_names):
        # create data frames from pickle files if not create pickle files
        sheets = {}
        for sheet_name in sheet_names:
            print('read excel files: ', path, ' - ',sheet_name)
            sheets[sheet_name] = pd.read_excel(path, 
                                               sheet_name=sheet_name, 
                                               header=0, 
                                               na_values='#### not defined ###', 
                                               keep_default_na=False)
        return sheets


def input_excel_database(excel_path, sheet):
    sheets = [sheet]
    
    df = Excel(excel_path, sheets).sheets[sheet]
    # strip all strings from excel database
    df.replace(r'(^\s+|\s+$)', '', regex=True, inplace=True)
    df.replace(r'\s+', ' ', regex=True, inplace=True)
    return df






ARC_LENGTH_ADJUST = 0.015
SCREEN_WIDTH = 500



def img_show(img_name, img):
    return
    if isinstance(img, list):
        for each_img in img:
            cv2.imshow(img_name, each_img)
    else:
        cv2.imshow(img_name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_img(img, dimension, axis='x'):
    dimensions = img.shape
    if axis=='x':
        fx = dimension/dimensions[0]
        fy = fx
    else:
        fy = dimension/dimensions[1]
        fx = fx

    img = cv2.resize(img, None, fx=fx, fy=fy)
    dimensions = img.shape
    return img


def find_contours(img):
    try: # if the img is BGR
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # select range of specific colors
        mask = mask_white(hsv)
        img = mask

        # convert image to gray scale. This will remove any color noise
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # use threshold to convert the image to black and white with less noise
        img = cv2.bilateralFilter(img, 11, 17, 17)
        img = cv2.Canny(img, 0, 100)        
        
        #img_show('',img)
    except Exception as e: # if others (gray, BW)
        pass
    
    # find all contours
    contours,h = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_bounding_box(img, contour, shape='rectangle'):
    '''
    draw a bounding box of the contour
    shape = 'rectangle' or 'circle'
    '''
    img = img.copy()

    if shape=='circle':
        center, radius = cv2.minEnclosingCircle(contour)
        cv2.circle(img, center, radius, (255,0,0),2)
    else:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    return img



def draw_all_contours(img, contours, edge_num=None, minimun_area=None):
    img_contour = img.copy()
    for cnt in contours:

        # shape of contour # 0.015 need to adjust to find the best number for rectangle
        approx = cv2.approxPolyDP(cnt,ARC_LENGTH_ADJUST*cv2.arcLength(cnt,True),True)

        # calculate area of contour
        area = cv2.contourArea(cnt)

        # check proper contours
        if edge_num==None: 
            pass
        else:
            if len(approx)==edge_num:
                pass
            else:
                continue

        if minimun_area==None:
            pass
        else:
            if area>minimun_area:
                pass
            else:
                continue

        x,y,w,h = cv2.boundingRect(cnt)
        cv2.drawContours(img_contour,[cnt],0,(0,255,0),2)
        #cv2.putText(img_contour, str(len(approx)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        
    return img_contour



def find_screen_contour(img):
    '''
    detect white screen of the tool, the img trasfer should be a totally white screen (no content)
    the return value is a contour of the frame detected
    '''
    img = img.copy()

    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = mask_white(hsv) # detect white screen
    #img_show('mask', mask)
  
    # find contours
    contours = find_contours(mask)
    img = draw_all_contours(img,contours)
    img_show('xem contours de tim screen', img)
    screen_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]

    return screen_contour



def mask_white(hsv):
    mask = cv2.inRange(hsv, (0, 0, 1), (180, 20, 255)) # detect white screen
    return mask



def get_object_and_main_color(img_pil): 
    img = np.array(img_pil)
    img_temp = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    img = cv2.copyMakeBorder(img,30,30,30,30,cv2.BORDER_CONSTANT,value=[0,0,0])

    screen_contour = find_screen_contour(img)
    x,y,w,h = cv2.boundingRect(screen_contour)
    img = img[y+1:y+h-2, x+1:x+w-2]

    contours = find_contours(img_temp)
    img1 = draw_all_contours(img, contours, edge_num=None, minimun_area=None)
    img_show('find all contours',img1)
    # Concatenate all contours
    contour = np.concatenate(contours)
    img2 = draw_all_contours(img, [contour], edge_num=None, minimun_area=None)
    img_show('find countour image',img2)
    x,y,w,h = cv2.boundingRect(contour)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    img = img[y:y+h, x:x+w]
    #save_img_path = re.sub(r'\..*$', '_temp.jpg',img_path)
    #cv2.imwrite(save_img_path, img)

    main_color_rgb = find_main_color(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_show('object', img)
    im_pil = Image.fromarray(img)
    return im_pil, main_color_rgb


def find_main_color(img):
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # detect white
    mask = mask_white(hsv)
    mask = cv2.bitwise_not(mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    main_color_rgb = cv2.mean(img, mask=mask)[:3]
    main_color_rgb = tuple(int(x) for x in main_color_rgb)

    return main_color_rgb





def get_stardard_photo_names(word):
        standard_photo_names = [word + ' - img 1.jpg',
            word + ' - img 1.jpg',
            word + ' - img 1 - co phien am.jpg',
            word + ' - img 2.jpg',
            word + ' - img 3.jpg',
            word + ' - img 4.jpg',
            word + ' - img 5.jpg']
        return standard_photo_names

def check_missing_photos(df, passed_photo_names):
    for index, row in df.iterrows():
        word  = row['word'].lower()
        print(word)
        word_passed_photos = filter(lambda x: re.search(r'^.*'+word+'.*$', x, re.I) , passed_photo_names)
        standard_photo_names = get_stardard_photo_names(word)
        missing_photos = list(set(standard_photo_names) - set(word_passed_photos))
        print(missing_photos)


def resize_photo(src_path, des_path, border=False, add_word=''):
    BASE_WIDTH = 1400
    BASE_HEIGHT = 700
    BASE_FRAME = [BASE_WIDTH, BASE_HEIGHT]
    BASE_EDGE_RATIO = BASE_WIDTH/BASE_HEIGHT
    OFFSET = 30
    MAXIMUM_OBJECT_AREA = 500000
    with open(src_path, 'r+b') as f:
        with Image.open(f) as image:
            if border==True:
                # get object, do it 2 times to reduce noice
                obj, main_color_rgb = get_object_and_main_color(image)
                print('object size 1: ', obj.size)
                obj = ImageOps.expand(obj,border=OFFSET,fill='white') # add white border to object to repeat the object detectiion
                obj, main_color_rgb = get_object_and_main_color(obj)
                print('object size 2: ', obj.size)

                # resize object to fit the frame
                obj_size = obj.size
                edge_ratio = obj_size[0]/obj_size[1]
                if edge_ratio < BASE_EDGE_RATIO:
                    resize_ratio = BASE_HEIGHT/obj_size[1]
                    new_obj_size = (int(obj_size[0]*resize_ratio), int(obj_size[1]*resize_ratio - OFFSET*2))
                else:
                    resize_ratio = BASE_WIDTH/obj_size[0]
                    new_obj_size = (int(obj_size[0]*resize_ratio - OFFSET*2), int(obj_size[1]*resize_ratio))
                
                # keep the object area smaller than a specific number
                object_area = new_obj_size[0]*new_obj_size[1]
                if object_area > MAXIMUM_OBJECT_AREA:
                    resize_ratio = MAXIMUM_OBJECT_AREA/object_area
                    new_obj_size = (int(new_obj_size[0]*resize_ratio), int(new_obj_size[1]*resize_ratio))


                # resize object to new size
                obj = obj.resize(new_obj_size)
                obj = ImageOps.expand(obj,border=OFFSET,fill='white')
                print('object size 3: ', obj.size)

                # handle adding word and IPA
                if add_word=='':
                    
                    slide = Image.new('RGB', BASE_FRAME, color = 'white')
                    slide.paste(obj, (int((BASE_WIDTH - obj.size[0])/2), (int((BASE_HEIGHT- obj.size[1])/2))))

                else:
                    img_word = draw_word(add_word, main_color_rgb)


                    # the case when there is still horizontal space 
                    if edge_ratio < BASE_EDGE_RATIO:
                        obj_width = obj.size[0]
                        word_width = img_word.size[0]
                        space_width = int((BASE_WIDTH - (obj_width + word_width))/3)
                        slide = Image.new('RGB', BASE_FRAME, color = 'white')

                        slide.paste(img_word, (space_width, int((BASE_HEIGHT - img_word.size[1])/2)))
                        slide.paste(obj, (space_width*2 + word_width, int((BASE_HEIGHT - obj.size[1])/2)))
                    else:
                        obj_height = obj.size[1]
                        word_height = img_word.size[1]
                        space_height = int((BASE_HEIGHT - (obj_height + word_height))/3)
                        slide = Image.new('RGB', BASE_FRAME, color = 'white')

                        slide.paste(img_word, (int((BASE_WIDTH - img_word.size[0])/2), space_height))
                        slide.paste(obj, (int((BASE_WIDTH - obj.size[0])/2), space_height*2 + word_height))

            else:
                slide = resizeimage.resize_cover(image, BASE_FRAME)

            slide = slide.convert('RGB')
            slide.save(des_path, image.format)
    slide.show()


def draw_word(text, main_color_rgb):
    SPACE = 40
    phonics = '/{}/'.format(ipa.convert(text))
    
    word_font = ImageFont.truetype('Myriad Pro Bold.ttf', 70)
    word_size = word_font.getsize(text)

    phonics_font = ImageFont.truetype('CALIBRI.TTF', 50)
    phonics_size = phonics_font.getsize(phonics)

    size = (max(word_size[0],phonics_size[0]), word_size[1]+phonics_size[1]+SPACE)

    img = Image.new('RGBA', size, (255,255,255,255))
    draw = ImageDraw.Draw(img, 'RGBA')

    draw.text((int((size[0]-word_size[0])/2), 0), 
            text = text, 
            fill=main_color_rgb, 
            font=word_font, 
            stroke_width=2, 
            stroke_fill='black')
    draw.text((int((size[0]-phonics_size[0])/2), int((size[1]-word_size[1])/2)+SPACE), 
            text = phonics, 
            fill=main_color_rgb, 
            font=phonics_font,
            stroke_width=1, 
            stroke_fill='black')

    #im_pil.save(img_path, image.format)
    #img.show()
    return img



def create_auto_resize_photos(df, raw_photo_names, 
            raw_photo_folder_path, 
            auto_resize_folder_path, 
            passed_photo_names):
    for index, row in df.iterrows():
        word  = row['word'].lower()
        if not row['is_passed']:
            word_raw_photo_names = list(filter(lambda x: re.search(r'^.*'+word+'.*$', x, re.I) , raw_photo_names))

            for photo_name in word_raw_photo_names:
                if '_temp.' in photo_name: continue

                # check if the photo is "1" => repeat 2 times, one for norma slide, one for word and phonics added
                if '1' in photo_name: 
                    loop_number = 2
                else:
                    loop_number = 1

                # for loop in case the photo is 1
                for loop in range(0,loop_number):


                    src_path = raw_photo_folder_path  + photo_name

                    # photo name in destination
                    if loop==2: # if loop the photo 1 the second time (time for word and phonics added)
                        photo_name = re.sub(r'\..*$', '1.jpg', photo_name.replace(word, word + ' - img 1 - co phien am'))
                    else:
                        photo_name = re.sub(r'\..*$', '.jpg', photo_name.replace(word, word + ' - img'))

                    # create destination path from photo name and folder path
                    des_path = auto_resize_folder_path + photo_name


                    # add white border (padding) to photo 1 and 2
                    if any([str(x) in des_path for x in [1,2]]):
                        add_border = True
                    else:
                        add_border = False

                    # add word and phonics to photo 1 - 2nd time
                    if loop==2:
                        add_word = word.capitalize()
                    else:
                        add_word = ''

                    # check if the photo is already in QC passed folder
                    if photo_name not in passed_photo_names:
                        try:
                            resize_photo(src_path, des_path, add_border, add_word)
                        except Exception as e:
                            print('>>>>>>>>>>>>>>>>> ERROR - ', word, ': ', e)
                            raise e
                    else:
                        print(photo_name, ': already in QC passed')

        print(word, '=> done')

