# ML-Project

This repository contains the code for my project for the course CSE-60625 Machine Learning.

The project explores the relation between text and style of images. In particular, it tries to learn a mapping between sentence embeddings and AdaIN parameters for style transfer.

The code requires the MS-COCO dataset. You can download the dataset here: https://cocodataset.org/#download 

The style text-image pair dataset loader requires the val2014 split of the MS-COCO dataset as it utilizes the Senticap dataset. The Senticap dataset present in  the 'senticap_dataset.json' file contains sentiment-polarized captions for images in val2014 of the MS-COCO dataset.

To train the generator-based architecture, run:

> python train_ps_generator.py --content_dir <content_image_path> --style_dir <style_dir_path>

The <style_dir_path> should be the val2014 split of MS-COCO. The <content_image_path> can actually be any large enough dataset of images, I use the train2014 split of MS-COCO.

The weights of the model can be found here: https://drive.google.com/drive/folders/19_douwjnBN9GvCOPbWDzVYuNjkJ3Q57D?usp=sharing

To stylize an image using text, run:

> python stylize_image_with_text.py --content_image <content_image_path> --style_text <style_text> --weight <weight_path>

You can use experiments_generator/iter_87500.pth as the weight, content.png as the content image, and 'The beautiful blue sky soothes me' as the style text to generate the image given in the report. 
