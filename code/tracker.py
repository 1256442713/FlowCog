from typing import List
import sys
import os
import numpy as np
import torch
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import time
import cv2
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepsort'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CenterTrack/src'))

from deepsort.yolo3_deepsort import Runner as Runner_deepsort_v3
from deepsort.yolo4_deepsort import Runner as Runner_deepsort_v4
from tracker_api import Runner as Runner_centertrack
#from FairMOT.src.FairMOT import Runner as Runner_FairMOT
import scene
# code liangxie
from attribute_analyze.analyze import BoxManager
from collections import Counter


#reid
sys.path.append(r"./vehicle_reid/algappspy_master")
sys.path.append(r"./vehicle_reid/reidframwork_master/reidframwork_master")
from vehicle_reid.algappspy_master.CarFeatureExtractor.CarFeatureExtractor import CarFeatureExtractor


def get_st_tracker(stream, args):
    """Get the short-term tracker"""
    run_deepsort_flag = 0
    if args.tracker == "yolov3_deepsort":
        tracker = Runner_deepsort_v3(stream)
    elif args.tracker == "yolov4_deepsort":
        tracker = Runner_deepsort_v4(stream)
        run_deepsort_flag = 1
    elif args.tracker == 'CenterTrack':
        tracker = Runner_centertrack(stream, args.track_thresh, args.load_model, args.gpus)
    else:
        raise NotImplementedError('{} is not supported now!'.format(args.tracker))
    return tracker, run_deepsort_flag

def takeThird(elem):
    return elem[2]


class Tracker:
    def __init__(self, stream, routes, args):
        # 启动deepsort标志：初始值为0，启动则设为1，
        self._st_tracker, self._run_deepsort_flag = get_st_tracker(stream, args)
        self._routes = routes
        self._traffic_scene = None if routes is None else scene.TrafficScene(routes)
        self._featExtractor = CarFeatureExtractor(r"./vehicle_reid/reidframwork_master/reidframwork_master/config/vehicleExtractor.yaml")
        self._save_dir = r"./vehicle_reid/save_reid_image/"
        self._reid_box_manager = BoxManager()
        self._previous_track_ids = []
        self._reid_threshold = 0.60
        self._filter_image_size_threshold = 45
        self.count_traj_map = 0
        self.count_ = 0

    def run(self, frame: np.ndarray, frame_idx: int):
        finished_trajs, outputs, lost_ids = self._st_tracker.run(frame)
        #print('after run tracker {}: {}'.format(frame_idx, outputs))
        assert isinstance(finished_trajs, dict)
        # lost_ids = [id for id in lost_ids if id not in finished_trajs]
        # add yolov4, do not output lostid
        counts, counted_traj_routes, matches = self._match_traj_route(finished_trajs, outputs,
                                                                      lost_ids, frame_idx)

        # calculate the count and counted_traj_routes of all
        #track_ids_list = [i for i in counted_traj_routes]
        #print("***** after _match_traj_route: chi: counts = ", np.sum(np.array(counts)))
        #print("***** after _match_traj_route: chi: counted_traj_routes = ", len(track_ids_list))

        # TODO ReID here
        true_matches_return = None
        if (not self._run_deepsort_flag) == 1:     # yolov4 do not open this
            print("*****  before reid ****************")
            # 先维护reid的表，之前是放到match外面
            if not len(outputs) == 0:
                # update maintain list
                self._reid_box_manager.set_cur_frame(frame)
                track_ids = outputs[:, 4]
                maintain_ids = set(track_ids).intersection(set(self._previous_track_ids))  # 维护
                new_ids = set(track_ids) - maintain_ids  # 新增
                disappear_ids = set(self._previous_track_ids) - maintain_ids  # 消失

                # disappear id
                for dis_id in disappear_ids:
                    self._reid_box_manager.move_tid_to_disappear_dict(
                        dis_id)  # for disappeared id, delete from the id-prediction dictionary

                # new id
                for output in outputs:
                    if output[4] in new_ids:  # for new id, put in new id queue, init result as None
                        self._reid_box_manager.add_tid_bimg(output)
                    elif output[4] in maintain_ids:  # for maintain id, search the according prediction result
                        self._reid_box_manager.update_tid_bimg(output)
                    else:
                        warnings.warn("Warning: track id not in new_ids and maintain_ids")
                self._previous_track_ids = track_ids

            else:
                self._previous_track_ids = []

        if matches:  # not None and []
            # reid feature Extractor  and save similarity
            # accord the max similarity choose the delete id (re-rank)
            # every id has only similarity and compare with reid threshold
            similarity = []
            for index in range(len(matches)):
                lost_tid = matches[index][0]
                cur_tid = matches[index][1]
                #print('reid: lost_tid : {}, cur_tid : {}'.format(lost_tid,cur_tid))

                # judge lost_tid and cur_tid
                try:
                    assert lost_tid in list(self._reid_box_manager.disappear_tid_info.keys())
                    assert cur_tid in list(self._reid_box_manager.tid_bimg_info.keys())
                except Exception:
                    warnings.warn('lost_tid {} not in disappear dictionary \
                    or cur_tid {} not in tid_bimg_info dictionary.\n \
                    disapprear keys: {} \n \
                    tid_bimg keys: {}'.format(lost_tid,
                                             cur_tid,
                                             list(self._reid_box_manager.disappear_tid_info.keys()),
                                             list(self._reid_box_manager.tid_bimg_info.keys())))

                if lost_tid not in list(self._reid_box_manager.disappear_tid_info.keys()) or \
                    cur_tid not in list(self._reid_box_manager.tid_bimg_info.keys()):
                    # print('lost_tid {} not in disappear dictionary \
                    #                    or cur_tid {} not in tid_bimg_info dictionary.\n \
                    #                    disapprear keys: {} \n \
                    #                    tid_bimg keys: {}'.format(lost_tid,
                    #                                              cur_tid,
                    #                                              list(self._reid_box_manager.disappear_tid_info.keys()),
                    #                                              list(self._reid_box_manager.tid_bimg_info.keys())))
                    continue

                # get track_id lost_id box image
                lost_id_bimg = self._reid_box_manager.get_lost_tid(lost_tid)[1]
                cur_id_bimg = self._reid_box_manager.tid_bimg_info[cur_tid][1]

                #accord the image size to delete
                if (lost_id_bimg.shape[1] < self._filter_image_size_threshold and lost_id_bimg.shape[0] < self._filter_image_size_threshold) or \
                        (cur_id_bimg.shape[1] < self._filter_image_size_threshold and cur_id_bimg.shape[0] < self._filter_image_size_threshold):
                    #print("reid: lost_id_bimg size < 45 or track_id_bimg size < 45")
                    continue

                # reid extract feature
                lost_id_bimg1 = lost_id_bimg[np.newaxis, ...]
                cur_id_bimg1 = cur_id_bimg[np.newaxis, ...]
                feat_q = self._featExtractor.do_alg({"image": lost_id_bimg1})
                feat_g = self._featExtractor.do_alg({"image": cur_id_bimg1})
                cos_similary = cosine_similarity(feat_q["result"], feat_g["result"])
                if cos_similary[0][0] > self._reid_threshold:
                    similarity.append([lost_tid, cur_tid, cos_similary[0][0]])
                else:
                    #print("reid: cos_similary[0][0] small than the threshold ")
                    continue

                # TODO save image to see if the image is similarity
                if not os.path.exists(self._save_dir):
                    os.mkdir(self._save_dir)
                lost_image_name = self._save_dir + "frame_"+ str(frame_idx) + "_lost_id_" + str(lost_tid) + "_track_id_" + str(cur_tid) + "_similarity_"+ str(cos_similary[0][0]) + "_lost_id_bimg.jpg"
                cv2.imwrite(lost_image_name, lost_id_bimg)
                track_image_name = self._save_dir + "frame_" + str(frame_idx) + "_track_id_" + str(cur_tid) + "_lost_id_" + str(lost_tid) + "_similarity_"+ str(cos_similary[0][0]) + "_track_id_bimg.jpg"
                cv2.imwrite(track_image_name, cur_id_bimg)

            #accord max similarity to delete the same element in lost id and track id
            maximum_match = []
            similarity.sort(key=takeThird, reverse=True)
            #print("reid: similarity sort:", similarity)
            while len(similarity)>0:
                maximum_match.append(similarity[0])
                to_remove = []
                id_lost = similarity[0][0]
                id_track = similarity[0][1]
                for j in range(0, len(similarity)):
                    if id_lost == similarity[j][0] or id_track == similarity[j][1]:
                        to_remove.append(similarity[j])
                for k in to_remove:
                    similarity.remove(k)
            #print("reid: maximum_match=",maximum_match)
            # accord max similarity left to assciated
            true_matches = []

            for item in range(len(maximum_match)):
                true_lost = maximum_match[item][0]
                true_track = maximum_match[item][1]
                self._traffic_scene.associate(true_lost, true_track)
                true_matches.append([true_lost, true_track])
                # TODO not match what time to delete
                self._reid_box_manager.del_dis_tid(true_lost)
            self._st_tracker.associate(true_matches)
            true_matches_return = true_matches

        # accord the finish traj to delete disappear ids
        for finished_id in finished_trajs.keys():
            # print('after reid: finished_trajs tids: {}'.format(list(finished_trajs.keys())))
            if finished_id in list(self._reid_box_manager.disappear_tid_info.keys()):
                self._reid_box_manager.del_dis_tid(finished_id)  # for disappeared id, delete from the id-prediction dictionary

        # verify the delete
        # print('after reid: tid_bimg_info tids: frame_idx= {},{}'.format(frame_idx, list(self._reid_box_manager.tid_bimg_info.keys())))
        # print('after reid: disappear_info tids: frame_idx= {}, {}'.format(frame_idx, list(self._reid_box_manager.disappear_tid_info.keys())))

        return counts, counted_traj_routes, finished_trajs, outputs, true_matches_return

    @property
    def routes(self):
        return self._routes

    def update_routes(self, routes):
        if routes is None:
            return
        self._routes = routes
        if self._traffic_scene is None:
            self._traffic_scene = scene.TrafficScene(routes)
        else:
            self._traffic_scene.update_routes(routes)

    def _match_traj_route(self, finished_trajs, outputs: np.ndarray, lost_ids: List[int],
                          frame_idx: int):
        if self._traffic_scene is None:
            return None, None, None

        for finish_id in finished_trajs:
            self._traffic_scene.delete(finish_id)

        lost_ids = [track_id for track_id in lost_ids if self._traffic_scene.exists(track_id)]

        # mark some lost track ids to deleted by means of route check
        delete_ids = []
        for track_id in lost_ids:           # yolov4 the lostid is empty
            route_idx = self._traffic_scene.assigned_route(track_id)
            if route_idx == -1:
                # lost before counted, so it can be safely deleted
                delete_ids.append(track_id)
            elif self._traffic_scene.landmark_idx(track_id) >= \
                    self._traffic_scene.num_landmark_at(route_idx) - 1:
                # lost near the end of the route, it can also be safely deleted
                delete_ids.append(track_id)
        if (not self._run_deepsort_flag) == 1:       # yolov4 do not run this, centertrack run
            finished_trajs.update(self._st_tracker.mark_delete(delete_ids))

        track_box_vec = scene.TrackBoxVec()
        for track in outputs:
            track_id = track[4]
            box = scene.Rect(track[0], track[1], track[2], track[3])
            track_box_vec.append(scene.TrackBox(track_id, frame_idx, box))
        self._traffic_scene.receive_trackbox(track_box_vec)

        counts = [self._traffic_scene.count_at(i) for i in range(self._traffic_scene.num_routes())]
        assigned_traj_ids = self._traffic_scene.assigned_traj_ids()

        # find lost track id and tracked id pairs that may be ReID
        matches = []
        for track in outputs:
            track_id = track[4]
            route = self._traffic_scene.assigned_route(track_id, True)
            if route == -1:
                continue
            landmark = self._traffic_scene.first_landmark_idx(track_id)
            for track_id_lost in lost_ids:      # yolov4 dot not run here, match is empty
                route_lost = self._traffic_scene.assigned_route(track_id_lost)
                # TODO the condition `route_lost == -1` may be removed
                if route_lost == -1 or route_lost != route:
                    continue
                landmark_lost = self._traffic_scene.landmark_idx(track_id_lost)
                if landmark_lost <= landmark:
                    matches.append((track_id_lost, track_id))

        #print("***** after _match_traj_route: every frame: counts = ", counts)
        #print("***** after _match_traj_route: every frame: assigned_traj_ids = ", assigned_traj_ids)

        return counts, assigned_traj_ids, matches

    def release(self):
        return self._st_tracker.release()
