import argparse
import subprocess as sp
import queue
import sys
import os
import time
import warnings
from collections import OrderedDict
import cv2
import numpy as np
import datetime

parser = argparse.ArgumentParser('--- Multiple object tracking and counting ---')
parser.add_argument('--data_path', default=None,
                    help="video path or image sequences directory (which contains the 'frames' sub-directory)")
parser.add_argument('--tracker', default='CenterTrack', choices=('CenterTrack', 'yolov3_deepsort', 'yolov4_deepsort', 'FairMOT'),
                    help='tracker method to use, default: CenterTrack')
parser.add_argument('--track_thresh', default=0.4, type=float, help='tracker output threshold')
parser.add_argument('--load_model', default=None, help='tracker model path')
parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
parser.add_argument('--track_only', action='store_true')
parser.add_argument('--save_video', action='store_true')
parser.add_argument('--save_dir', default=None)
parser.add_argument('--attr_step', type=int, default=5)
parser.add_argument('--load_routes', default=None, help='routes path')
args = parser.parse_args()

# setup 'CUDA_VISIBLE_DEVICES' environment before import torch
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import torch
import torch.multiprocessing as mp
from image_stream import VideoIO
import scene
from transfer import sender, receiver
from drawer import Drawer
from attribute_analyze.analyze import AttrAnalysis, BoxManager
from tracker import Tracker
from scene_understanding import traj_cls_process
from output import organize_results


def send_traj(sender, finished_trajs):
    for k, v in finished_trajs.items():
        track_id = k
        rects = scene.RectVec()
        for bbox in v:
            rects.append(scene.Rect(bbox[0], bbox[1], bbox[2], bbox[3]))
        traj = scene.Traj(track_id, rects)
        sender.send_with_byte('trajs', traj.encode())


class AttributesProcess(mp.Process):

    def __init__(self, queue_track, queue_draw, event_track, event_draw, device, step_k):
        super(AttributesProcess, self).__init__()
        self._queue_track = queue_track
        self._queue_draw = queue_draw
        self._device = device
        self._step_k = step_k
        self._event_track = event_track
        self._event_draw = event_draw

    def run(self):
        attr_model_weight = r'./attribute_analyze/save/ckp_63_resnet34_t02_fl_d07_cp.pth'
        attr_label_info_file = r'./attribute_analyze/new_attr_info.json'
        attr_tarcode_use_file = r'./attribute_analyze/tarcode_use.cfg'
        cn2en_file = r'./attribute_analyze/CN_to_EN.json'
        with torch.cuda.device(self._device):
            attr_analyzer = AttrAnalysis(weight_path=attr_model_weight,
                                         label_info_file=attr_label_info_file,
                                         tarcode_use_file=attr_tarcode_use_file,
                                         cn2en_file=cn2en_file)
            attr_analyzer.data_model_prepare()
            box_manager = BoxManager()

        previous_track_ids = []
        flag = 0
        while True:
            items = self._queue_track.get()
            flag += 1
            if items is None:
                # inform counting function of the end
                self._queue_draw.put(None)
                break
            img, frame_idx, timestamp, outputs, true_matches, counts, counted_traj_routes, routes, routeid_lanemark_map = items
            frame = img.numpy()
            # frame = np.random.random((1200, 1600, 3))
            attr_result = []
            if (not args.track_only) and len(outputs) > 0:
                with torch.cuda.device(self._device):
                    # return results to count
                    for output in outputs:
                        if output[4] in list(box_manager.tid_bimg_info.keys()):
                            attr_result.append(box_manager.get_tid_pred(output[4]))
                        else:
                            attr_result.append(None)
                    self._queue_draw.put((img.clone(), frame_idx, timestamp, outputs, true_matches, routes,
                                          attr_result,
                                          counted_traj_routes, counts, routeid_lanemark_map))
                    # process bounding boxs
                    box_manager.set_cur_frame(frame)
                    track_ids = outputs[:, 4]
                    maintain_ids = set(track_ids).intersection(set(previous_track_ids))
                    new_ids = set(track_ids) - maintain_ids
                    disappear_ids = set(previous_track_ids) - maintain_ids

                    for dis_id in disappear_ids:
                        box_manager.del_tid_bimg(dis_id)  # for disappeared id, delete from the id-prediction dictionary

                    for output in outputs:
                        if output[4] in new_ids:  # for new id, put in new id queue, init result as None
                            box_manager.add_tid_bimg(output)
                        elif output[4] in maintain_ids:  # for maintain id, search the according prediction result
                            box_manager.update_tid_bimg(output)
                        else:
                            warnings.warn("Warning: track id not in new_ids and maintain_ids")

                    # k frame passed, attribute analyze model run
                    if flag % self._step_k == 0:
                        input_tid, input_data = box_manager.get_update_data()
                        if not input_tid:  # input_tid is empty
                            continue
                        preds = attr_analyzer.run(input_data)
                        box_manager.set_tid_preds(input_tid, preds)
                        flag = 0

                    previous_track_ids = track_ids
            else:
                attr_result = []
                previous_track_ids = []
                self._queue_draw.put((img.clone(), frame_idx, timestamp, outputs, true_matches, routes,
                                      attr_result,
                                      counted_traj_routes, counts, routeid_lanemark_map))

            # we need to delete tensors to allow producer (tracking process)
            # collect them properly
            del img

        # wait drawing process done
        self._event_draw.wait()
        # notice caller its end
        self._event_track.set()


class DrawerProcess(mp.Process):

    def __init__(self, queue_draw, queue_results, event_draw, event_output, width, height):
        super(DrawerProcess, self).__init__()

        self._queue_draw = queue_draw
        self._queue_results = queue_results
        self._event_draw = event_draw
        self._event_output = event_output
        self._width = width
        self._height = height

        self._drawer = Drawer(routes=None)

    def run(self):
        print('Start displaying')
        push_url = 'rtmp://192.168.202.92/myapp/gtf_demo'
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', "{}x{}".format(self._width, self._height),
                   '-r', '6',
                   '-i', '-',
                   '-rtsp_transport', 'tcp',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-f', 'flv',
                   push_url]
        p = sp.Popen(command, stdin=sp.PIPE)

        fp = open(os.path.join(args.save_dir, "counts.txt"), "w")
        while True:
            items = self._queue_draw.get()
            if items is None:
                self._queue_results.put(None)
                fp.close()
                p.kill()
                break
            img, frame_idx, timestamp, outputs, true_matches, routes, attr_result, counted_traj_routes, counts, routeid_lanemark_map \
                = items
            frame = img.numpy()
            if counted_traj_routes is None:
                assigned_traj_ids = None
            else:
                assigned_traj_ids = set(k for k in counted_traj_routes)
            frame = self._draw(routes, assigned_traj_ids, counts, frame, outputs, attr_result, routeid_lanemark_map)
            p.stdin.write(frame.tostring())
            if args.save_video:
                # TODO save video
                cv2.imwrite(os.path.join(args.save_dir, '%06d.jpg' % frame_idx), frame)
            if routes is not None:  # write counting results in file
                line = "%12d:" % int(timestamp)
                assert len(counts) == len(routes)
                for route, count in zip(routes, counts):
                    line += "%6d:%-6d " % (route.id, count)
                line += "\n"
                fp.write(line)

            # put results in queue_results
            res_item = (frame_idx, timestamp,
                        outputs, true_matches, routes, attr_result,
                        counted_traj_routes, counts, routeid_lanemark_map)

            self._queue_results.put(res_item)

            # we need to delete tensors to allow producer (attribute process)
            # collect them properly
            del img

        # wait output process done
        self._event_output.wait()
        # notice caller its end
        self._event_draw.set()

    def _draw(self, routes, assigned_traj_ids, counts, frame, outputs, attrs_pre_names, routeid_lanemark_map):
        self._drawer.routes = routes
        if assigned_traj_ids is not None:
            self._drawer.cross_track_ids = assigned_traj_ids
        self._drawer.update_trajs(outputs)
        self._drawer.draw_routes(frame, routeid_lanemark_map)
        if args.track_only:
            self._drawer.draw_bboxes_track(frame, outputs)
        else:
            frame = self._drawer.draw_bboxes_track_attrs(frame, outputs, attrs_pre_names)
        self._drawer.draw_trajs(frame)
        self._drawer.draw_counts(frame, counts, routeid_lanemark_map)
        return frame


class OutputProcess(mp.Process):
    def __init__(self, queue_results, event_draw, event_output):
        super(OutputProcess, self).__init__()
        self._queue_results = queue_results
        self._event_draw = event_draw
        self._event_output = event_output

    def run(self):
        print("Start receiving results")
        while True:
            items = self._queue_results.get()
            if items is None:
                break

            frame, frame_idx, timestamp,\
            outputs, true_matches, routes, attr_result,\
            counted_traj_routes, counts, routeid_lanemark_map = items

            # organize the results
            results = organize_results(items)

            # print("-----------------------------\n"
            #       "frame_idx: {}\n"
            #       "timestamp: {}\n"
            #       "outputs: {}\n"
            #       "true_matches: {}\n"
            #       "routes: {}\n"
            #       "attr_result: {}\n"
            #       "counted_traj_routes: {}\n"
            #       "counts: {}\n"
            #       "routeid_lanemark_map: {}".format(
            #         frame_idx, timestamp,
            #         outputs, true_matches, routes, attr_result,
            #         counted_traj_routes, counts, routeid_lanemark_map))

            print("--------------------------------")
            print(results)

        self._event_output.set()


def main(queue_track, event_track, stream, tracker):
    decode_responses = False
    sender_ = sender.SenderUtilser(
        host='192.168.202.90',
        password='pcl1305',
        auto_delete_limit=2000)
    receiver_ = receiver.ReceiverUtils(
        host='192.168.202.90',
        password='pcl1305',
        decode_responses=decode_responses)

    # set routes understanding classifier
    traj_clser = traj_cls_process.traj_cls()
    traj_clser.set_traj_id_name_map(r'./scene_understanding/traj_id_name_map_file.json')
    traj_clser.init_model(r'./scene_understanding/classifier.pkl')
    routeid_lanemark_map = None

    if args.load_routes is not None:
        assert os.path.exists(args.load_routes)
        routes = scene.load_route(args.load_routes)
        tracker.update_routes(routes)
        # routes understanding
        traj_clser.set_input(routes)
        routeid_lanemark_map = traj_clser.run()
        print('--------- routeid_lanemark_map ----------')
        print(routeid_lanemark_map)

    frame_idx = 0
    for timestamp, frame in stream:
        # check arrival of routes and then update traffic scene if true
        ret_list = receiver_.reveive_data(key='routes', limit_count=1)
        if len(ret_list) > 0:
            print('==> update routes')
            routes = scene.RouteVec()
            routes_py = scene.decode_routes(ret_list[0])
            for route_py in routes_py:
                routes.append(route_py.route)
            tracker.update_routes(routes)
            # routes understanding
            traj_clser.set_input(routes)
            routeid_lanemark_map = traj_clser.run()

        counts, counted_traj_routes, finished_trajs, outputs, true_matches = tracker.run(frame, frame_idx)

        # print("track time: {:.3f}".format((time.time() - start) * 1000))

        # step 1: send finished trajectory to server module
        assert isinstance(finished_trajs, dict)
        send_traj(sender_, finished_trajs)

        # step 2: put tracking results to the attributes extraction task
        # Note: send Pytorch Tensor which is efficient due to shared-memory;
        # only CPU Tensor are tested!!!
        img = torch.from_numpy(frame)
        # img.share_memory_()
        queue_track.put([img, frame_idx, timestamp, outputs, true_matches, counts, counted_traj_routes,
                         tracker.routes, routeid_lanemark_map])

        frame_idx += 1

    # send remaining finished trajectory
    finished_trajs = tracker.release()
    send_traj(sender_, finished_trajs)

    # notice the end of main thread
    queue_track.put(None)
    event_track.wait()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    queue_track = mp.Queue()
    queue_draw = mp.Queue()
    queue_results = mp.Queue()
    event_track = mp.Event()
    event_draw = mp.Event()
    event_output = mp.Event()

    if args.save_video:
        assert args.save_dir is not None
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    stream = VideoIO()
    stream.open(args.data_path)
    tracker = Tracker(stream, None, args)

    width = stream.im_width
    height = stream.im_height

    # launch attributes extraction process
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cuda:0')
    attr_task = AttributesProcess(queue_track, queue_draw, event_track, event_draw, device,
                                  args.attr_step)
    attr_task.daemon = True
    attr_task.start()

    # launch drawing and streaming process
    draw_task = DrawerProcess(queue_draw, queue_results, event_draw, event_output, width, height)
    draw_task.daemon = True
    draw_task.start()
    # display_thread = threading.Thread(target=display)
    # display_thread.start()

    output_task = OutputProcess(queue_results, event_draw, event_output)
    output_task.daemon = True
    output_task.start()

    # main thread
    main(queue_track, event_track, stream, tracker)

    # join all sub-processes
    attr_task.join()
    draw_task.join()
    output_task.join()
