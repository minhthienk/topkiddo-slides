from auto_slides_lib import *


'''
# THE EXCEL FILE FOR MANAGING WORDS
excel_path = '/run/media/minhthienk/Data/TOPKIDDO/Data.xlsx'
df = input_excel_database(excel_path, 'data')

# PATH WHERE RAW IMAGES ARE STORED
raw_photo_folder_path = '/run/media/minhthienk/Data/TOPKIDDO/raw photos/HINH DE CHAY AUTO/Unit 4/'
raw_photo_paths = [f for f in listdir(raw_photo_folder_path) if isfile(join(raw_photo_folder_path, f))]

# PATH WHERE PROCESSED SLIDES WILL BE STORED
auto_resize_folder_path = '/run/media/minhthienk/Data/TOPKIDDO/Auto Resize/Unit 4/'

# PATH WHERE ALREADY DONE AND CHECKED PASS SLIDES ARE STORED
qc_passed_photo_folder_path = '/run/media/minhthienk/Data/TOPKIDDO/QC Passed/'
passed_photo_paths = [f for f in listdir(qc_passed_photo_folder_path) if isfile(join(qc_passed_photo_folder_path, f))]

# RUN
create_auto_resize_photos(df, raw_photo_paths, raw_photo_folder_path, auto_resize_folder_path, passed_photo_paths)

'''


SHOW_IMAGE_FLAG = True
src_path = 'Toilet paper 1.jpg'
des_path = 'test.jpg'
add_border = True
add_word = 'whale'
resize_photo(src_path, des_path, add_border, add_word='coder')

