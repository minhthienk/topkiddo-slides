from auto_slides_lib import *



for i in range(0,1):
    print(i)


import sys
sys.exit()

excel_path = '/run/media/minhthienk/Data/TOPKIDDO/Data.xlsx'
df = input_excel_database(excel_path, 'data')

raw_photo_folder_path = '/run/media/minhthienk/Data/TOPKIDDO/raw photos/'
raw_photo_paths = [f for f in listdir(raw_photo_folder_path) if isfile(join(raw_photo_folder_path, f))]

auto_resize_folder_path = '/run/media/minhthienk/Data/TOPKIDDO/Auto Resize/'
qc_passed_photo_folder_path = '/run/media/minhthienk/Data/TOPKIDDO/QC Passed/'
passed_photo_paths = [f for f in listdir(qc_passed_photo_folder_path) if isfile(join(qc_passed_photo_folder_path, f))]


print(passed_photo_paths)
#create_auto_resize_photos(df, raw_photo_paths, raw_photo_folder_path, auto_resize_folder_path, passed_photo_paths)



'''

src_path = 'Coder 2.jpg'
des_path = 'test.jpg'
add_border = True
add_word = 'whale'
resize_photo(src_path, des_path, add_border, add_word='coder')

#todo:
#ti le hinh qua sat frame thi phair scale lai'''