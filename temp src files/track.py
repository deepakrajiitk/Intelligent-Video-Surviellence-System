import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms as T
import json
from PIL import Image
from datetime import timedelta



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
sys.path.append(str(ROOT/'Person_Attribute_Recognition_MarketDuke')) # add parm ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from trackers.multi_tracker_zoo import create_tracker
from Person_Attribute_Recognition_MarketDuke.net import get_model

######################################################################

#mySQL Database Settings
import mysql.connector

class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles Numpy types """

    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)

class Database:
    def __init__(self):
        # connecting to database
        self.mydb = mysql.connector.connect(
        host="localhost",
        user="deepak",
        password="deepak",
        database="cctv_database"
        )
        self.mydb.set_converter_class(NumpyMySQLConverter)
        self.mycursor = self.mydb.cursor()
        self.attributes = ["young", "teenager", "adult", "old", "backpack", "bag", "handbag", "clothes", "down", "up",
        "hair", "hat", "gender", "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgrey", "upblue", "upgreen", "downblack", "downwhite", "downpink",
        "downpurple", "downyellow", "downgrey", "downblue", "downgreen", "downbrown"]
        self.table_name = "cctv_table"
        self.threshold = 0.5
        # table is created one time only
        # self.create_main_table()

    def create_main_table(self):
        # query format
        table_creation_query = "create table if not exists "+ self.table_name+" (attributes text NOT NULL)";
        self.mycursor.execute(table_creation_query);
        # query = "insert into " +self.table_name+ " (attributes) values (%s)";
        for attribute in self.attributes:
            self.add_to_main_table_column("attributes", attribute)
        
    def add_to_main_table_column(self, columnName, value):
        query = "insert into " + self.table_name + " (" + columnName + ") " + " values (%s)"
        self.mycursor.execute(query, [value])
        self.mydb.commit();
    
    def main_table_insert(self, video_id):
        column_name = video_id
        check_query = "SELECT * FROM information_schema.COLUMNS WHERE TABLE_NAME = '{}' AND COLUMN_NAME = '{}'".format(self.table_name, column_name)
        self.mycursor.execute(check_query)
        result = self.mycursor.fetchone()
        if result is None:
            query1 = "ALTER TABLE {} ADD COLUMN {} INT".format(self.table_name, column_name)
            self.mycursor.execute(query1)

        for i in range(len(self.attributes)):
            query2 = "UPDATE {} SET {} = (SELECT COUNT(*) FROM {} WHERE {} > {}) WHERE attributes = '{}'".format(self.table_name, column_name, column_name, self.attributes[i], self.threshold, self.attributes[i])
            self.mycursor.execute(query2)

        self.mydb.commit()

    
    def create_video_table(self, table_name):
        table_creation_query = "CREATE TABLE if not exists "+table_name+" \
        (date VARCHAR(255) NOT NULL, \
        video_id VARCHAR(255) NOT NULL, \
        person_id VARCHAR(255) NOT NULL, \
        timeframe VARCHAR(255) NOT NULL, \
        young float NOT NULL, \
        teenager float NOT NULL, \
        adult float NOT NULL, \
        old float NOT NULL, \
        backpack float NOT NULL, \
        bag float NOT NULL, \
        handbag float NOT NULL, \
        clothes float NOT NULL, \
        down float NOT NULL, \
        up float NOT NULL, \
        hair float NOT NULL, \
        hat float NOT NULL, \
        gender float NOT NULL, \
        upblack float NOT NULL, \
        upwhite float NOT NULL, \
        upred float NOT NULL, \
        uppurple float NOT NULL, \
        upyellow float NOT NULL, \
        upgrey float NOT NULL, \
        upblue float NOT NULL, \
        upgreen float NOT NULL, \
        downblack float NOT NULL, \
        downwhite float NOT NULL, \
        downpink float NOT NULL, \
        downpurple float NOT NULL, \
        downyellow float NOT NULL, \
        downgrey float NOT NULL, \
        downblue float NOT NULL, \
        downgreen float NOT NULL, \
        downbrown float NOT NULL)"
        self.mycursor.execute(table_creation_query);
        self.mydb.commit()
    
    def video_table_insert(self, table_name, values):
        table_insertion_query = "insert into " +table_name+" (date, video_id, person_Id, timeframe, young, teenager, adult, old, \
            backpack, bag, handbag, clothes, down, up, hair, hat, \
            gender, upblack, upwhite, upred, uppurple, upyellow, \
            upgrey, upblue, upgreen, downblack, downwhite, downpink, \
            downpurple, downyellow, downgrey, downblue, downgreen, \
            downbrown) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.mycursor.execute(table_insertion_query, values)
        self.mydb.commit()


    # def insert(self, values):
    #     self.mycursor.execute(self.query, values)
    #     self.mydb.commit()

######################################################################

# PAR Settings
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

######################################################################

# PAR Model and Data
def load_network(network, dataset, par_model_name):
    save_path = os.path.join('./Person_Attribute_Recognition_MarketDuke/checkpoints', dataset, par_model_name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(src):
    # src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src

######################################################################

# PAR Inference
class predict_decoder(object):

    def __init__(self, dataset):
        with open('./Person_Attribute_Recognition_MarketDuke/doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./Person_Attribute_Recognition_MarketDuke/doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        # print(self.attribute_dict)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))

######################################################################


@torch.no_grad()
def run(
        date='01_01_2022',
        video_id='video_0',
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        image_path = "",
        dataset="market",
        backbone="resnet50",
        use_id=True,
        frame_skip=0,
        save_db=False #save to SQL database
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    video_id = source[:-4]
    

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load PAR model
    par_model_name = '{}_nfc_id'.format(backbone) if use_id else '{}_nfc'.format(backbone)
    num_label, num_id = num_cls_dict[dataset], num_ids_dict[dataset]
    par_model = get_model(par_model_name, num_label, use_id=use_id, num_id=num_id)
    par_model = load_network(par_model, dataset, par_model_name)
    par_model.eval()
    Dec = predict_decoder(dataset)

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # creating database object
    if save_db:
        db = Database()
        # creating video table
        db.create_video_table(video_id)

    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

        # skipping frames
        if(frame_skip!=0 and frame_idx%frame_skip!=0):
            continue;

        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            if is_seg:
                pred, proto = model(im, augment=augment, visualize=visualize)[:2]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            if is_seg:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            else:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # below part is to show red and bounding box over cropped image
                if is_seg:
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                        masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                        
                    # Mask plotting
                    # below code is used to apply red color to the exact entity
                    annotator.masks(
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                        255 if retina_masks else im[i]
                    )
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid or save_db:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bboxes, label, color=color)

                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop or save_db:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                # save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                                image_file_name = str(date)+"_"+video_id+"_"+names[c]+"_"+str(id)+"_"+str(frame_idx);
                                # print("============", type(save_dir))
                                save_dir2 = Path("../../../var/www/html/CCTV/php/images")
                                crop = save_one_box(bboxes.astype(np.float32), imc, file=save_dir2 / f'{video_id}' / f'{image_file_name}.jpg', BGR=True)
                                # getting attributes using PAR
                                # tracking only persons (class 0)
                                if(c==0 and save_db and len(crop)!=0):
                                    t6 = time_sync()
                                    crop_image = load_image(Image.fromarray(crop[..., ::-1]))
                                    if not use_id:
                                        out = par_model.forward(crop_image)
                                    else:
                                        out, _ = par_model.forward(crop_image)
                                    values = list(out.numpy()[0])
                                    values.insert(0, str(frame_idx))
                                    values.insert(0, str(id))
                                    values.insert(0, str(video_id))
                                    values.insert(0, str(date))
                                    # print(values)
                                    # adding values to the database
                                    db.video_table_insert(video_id, values)
                                    # mycursor.execute(query, values)
                                    # mydb.commit()
                                    # pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
                                    # Dec.decode(pred)
                                    t7 = time_sync();
                
            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
                
            # Stream results
            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
    
    # inserting new video data on main table
    if save_db:
        db.main_table_insert(video_id)


    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--image_path', help='Path to test image for person attribute recognition(PAR')
    parser.add_argument('--dataset', default='market', type=str, help='dataset for PAR')
    parser.add_argument('--backbone', default='resnet50', type=str, help='model for PAR')
    parser.add_argument('--use-id', action='store_true', help='use identity loss for PAR')
    parser.add_argument('--frame-skip', default=0, type=int, help='number of frames to skip')
    parser.add_argument('--save-db', action='store_true', help='save to SQL database')
    opt = parser.parse_args()
    assert opt.dataset in ['market', 'duke']
    assert opt.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)