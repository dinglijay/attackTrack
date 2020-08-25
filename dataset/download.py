import json
import os
import wget, tarfile, zipfile
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
 
 
vot_2019_path = '/DataServer/tracking_dbs/vot2018/data/'  # object file
json_path = '/DataServer/tracking_dbs/dl/vot2018/description.json'  # vot 2019 json file
anno_vot = 'vot2018'  # vot2019 or vot2018 or vot2017
 
with open(json_path, 'r') as fd:
    vot_2019 = json.load(fd)
home_page = vot_2019['homepage']
 
for i, sequence in enumerate(vot_2019['sequences']):
    print('download the {} sequences'.format(i + 1))
    #
    annotations = sequence['annotations']['url']
    data_url = sequence['channels']['color']['url'].split('../../')[-1]
 
    name = annotations.split('.')[0]
    file_name = annotations.split('.')[0] + '.zip'
 
    down_annotations = os.path.join(home_page, anno_vot, 'longterm', annotations)  #main longterm
                                                                      #http://data.votchallenge.net/vot2019/longterm/ballet.zip
    down_data_url = os.path.join(home_page, data_url)
 
    image_output_name = os.path.join(vot_2019_path, name, 'color', file_name) #http://data.votchallenge.net/sequences/578a919e272d9993be8a27b0df4b0a12845f8e99e0fd04582d97f61a939a751cf0bafd68a8f21b4a7ef4fb6aeb43732e293b598cae5a8344ead827fdef279812.zip
    anno_output_name = os.path.join(vot_2019_path, name, file_name)
    out_dir = os.path.dirname(anno_output_name)
 
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)
    if os.path.exists(out_dir+'/groundtruth.txt') == True:
        pass
    else:
        # annotations download and unzip and remove it
        wget.download(down_annotations, anno_output_name)
        print('loading {} annotation'.format(name))
        # unzip
        file_zip = zipfile.ZipFile(anno_output_name, 'r')
        for file in file_zip.namelist():
            file_zip.extract(file, out_dir)
            print('extract annotation {}/{}'.format(name, file))
        file_zip.close()
        os.remove(anno_output_name)
        print('remove annotation {}.zip'.format(name))
 
    # image download and unzip ad remove it
    out_dir = os.path.dirname(image_output_name)
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)
    if os.path.exists(out_dir+'/00000001.jpg') == True:
        continue
    else:
        wget.download(down_data_url, image_output_name)
        print('loading {} sequence'.format(name))
 
        file_zip = zipfile.ZipFile(image_output_name, 'r')
        for file in file_zip.namelist():
            file_zip.extract(file, out_dir)
            print('extract image {}'.format(file))
        file_zip.close()
        os.remove(image_output_name)
        print('remove image file {}.zip'.format(name))
        print('sequence  {} Completed!'.format(i + 1))
