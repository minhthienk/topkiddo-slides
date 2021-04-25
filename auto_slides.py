
import re

from PIL import ImageFont
from PIL import ImageDraw 
from PIL import Image, ImageOps
from resizeimage import resizeimage

from os import listdir
from os.path import isfile, join
import os 
import pandas as pd

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






import numpy as np
import cv2

ARC_LENGTH_ADJUST = 0.015
SCREEN_WIDTH = 500



def img_show(img_name, img):
    if isinstance(img, list):
        for each_img in img:
            cv2.imshow(img_name, each_img)
    else:
        cv2.imshow(img_name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_img(img, width):
    dimensions = img.shape
    fx = width/dimensions[0]
    fy = fx
    img = cv2.resize(img, None, fx=fx, fy=fy)
    dimensions = img.shape
    return img


def find_contours(img):
    try: # if the img is BGR
        # convert image to gray scale. This will remove any color noise
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    # select range of specific colors
    mask = cv2.inRange(hsv, (0, 0, 1), (180, 10, 255)) # detect white screen
    #img_show('mask', mask)
  
    # find contours
    contours = find_contours(mask)
    img2 = draw_all_contours(img,contours)
    #img_show('xem contours de tim screen', img2)
    screen_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]

    return screen_contour


def get_object_and_main_color(img_pil): 
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
    img = cv2.copyMakeBorder(img,30,30,30,30,cv2.BORDER_CONSTANT,value=[0,0,0])

    screen_contour = find_screen_contour(img)
    x,y,w,h = cv2.boundingRect(screen_contour)
    img = img[y+1:y+h-2, x+1:x+w-2]

    contours = find_contours(img)
    # Concatenate all contours
    contour = np.concatenate(contours)
    #img2 = draw_all_contours(img, [contour], edge_num=None, minimun_area=None)
    x,y,w,h = cv2.boundingRect(contour)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    img = img[y:y+h, x:x+w]
    #save_img_path = re.sub(r'\..*$', '_temp.jpg',img_path)
    #cv2.imwrite(save_img_path, img)

    main_color_rgb = find_main_color(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil, main_color_rgb


def find_main_color(img):
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # select range of specific colors
    mask = cv2.inRange(hsv, (0, 0, 1), (180, 5, 255)) # detect white screen
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

def check_missing_photos(df, passed_photo_paths):
    for index, row in df.iterrows():
        word  = row['word'].lower()
        print(word)
        word_passed_photos = filter(lambda x: re.search(r'^.*'+word+'.*$', x, re.I) , passed_photo_paths)
        standard_photo_names = get_stardard_photo_names(word)
        missing_photos = list(set(standard_photo_names) - set(word_passed_photos))
        print(missing_photos)


def resize_photo(src_path, des_path, border=False, add_word=''):
    BASEFRAME = [1400, 700]
    OFFSET = 30
    with open(src_path, 'r+b') as f:
        with Image.open(f) as image:
            if border==True and add_word=='':
                image, main_color_rgb = get_object_and_main_color(image)
                img_with_border = ImageOps.expand(image,border=OFFSET,fill='white')
                cover = resizeimage.resize_contain(img_with_border, BASEFRAME)

            elif border==True and add_word!='':
                image, main_color_rgb = get_object_and_main_color(image)
                img_with_border = ImageOps.expand(image,border=OFFSET,fill='white')
                cover = resizeimage.resize_contain(img_with_border, BASEFRAME, bg_color=(255, 255, 255, 255))
                
                draw_word(add_word, cover, main_color_rgb)
                
            else:
                cover = resizeimage.resize_cover(image, BASEFRAME)

            cover = cover.convert('RGB')
            cover.save(des_path, image.format)


def draw_word(text, im_pil, main_color_rgb):
    files = os.listdir()
    font_names = [file for file in files if '.ttf' in file]
    font_name = font_names[0]
    font_size = 50
    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(im_pil)
    draw.text((0, 0), text, fill=main_color_rgb, font=font)
    #im_pil.save(img_path, image.format)
    im_pil.show()

font.getsize(text)[0] 
Gubbi 16
Noto serif light 8

def create_auto_resize_photos(df, raw_photo_paths, 
            raw_photo_folder_path, 
            auto_resize_folder_path, 
            passed_photo_paths):
    for index, row in df.iterrows():
        word  = row['word'].lower()
        if not row['is_passed']:
            word_raw_photo_paths = list(filter(lambda x: re.search(r'^.*'+word+'.*$', x, re.I) , raw_photo_paths))
            #print(word_raw_photo_paths)
            for photo_path in word_raw_photo_paths:
                if '_temp.' in photo_path: continue

                src_path = raw_photo_folder_path  + photo_path

                # photo name in destination
                photo_path = re.sub(r'\..*$', '.jpg', photo_path.replace(word, word + ' - img'))
                des_path = auto_resize_folder_path + photo_path

                if any([str(x) in des_path for x in [1,2]]):
                    add_border = True
                    add_word = 'Hello'
                else:
                    add_border = False
                    add_word = False

                if photo_path not in passed_photo_paths:
                    try:
                        resize_photo(src_path, des_path, add_border, add_word)
                    except Exception as e:
                        print('>>>>>>>>>>>>>>>>> ERROR - ', word, ': ', e)
                        raise e
                else:
                    print(photo_path, ': already in QC passed')

        print(word, '=> done')



excel_path = '/media/minhthienk/Data4/TOPKIDDO/Data.xlsx'
df = input_excel_database(excel_path, 'data')

raw_photo_folder_path = '/media/minhthienk/Data4/TOPKIDDO/raw photos/'
raw_photo_paths = [f for f in listdir(raw_photo_folder_path) if isfile(join(raw_photo_folder_path, f))]


auto_resize_folder_path = '/media/minhthienk/Data4/TOPKIDDO/Auto Resize/'
qc_passed_photo_folder_path = '/media/minhthienk/Data4/TOPKIDDO/QC Passed/'
passed_photo_paths = [f for f in listdir(qc_passed_photo_folder_path) if isfile(join(qc_passed_photo_folder_path, f))]


create_auto_resize_photos(df, raw_photo_paths, raw_photo_folder_path, auto_resize_folder_path, passed_photo_paths)



#resize_photo('3f6951d17224807ad935.jpg', '3f6951d17224807ad935zz.jpg', border=True)