import os, glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

image_path = os.path.join(os.getcwd(),'test_labels_xml')
xml_df = xml_to_csv(image_path)
xml_df.to_csv('test_labels.csv', index=None)

image_path = os.path.join(os.getcwd(),'test_labels_xml')
xml_df = xml_to_csv(image_path)
xml_df.to_csv('test_labels.csv', index=None)


image_path = os.path.join(os.getcwd(),'train_labels_xml')
xml_df = xml_to_csv(image_path)
xml_df.to_csv('train_labels.csv', index=None)

image_path = os.path.join(os.getcwd(),'validation_labels_xml')
xml_df = xml_to_csv(image_path)
xml_df.to_csv('validation_labels.csv', index=None)
