from __future__ import absolute_import
from __future__ import print_function

import os
import time
import argparse
from scene_understanding import predict

from transfer import receiver, sender
import scene

# TODO: get camera configuration from head
W = 1600
H = 1200
MAX_ITER = 1000
MAX_NUM_CLUSTER_TRAJS = 2000
ABNORMAL_PERCENT = 0.02
MIN_REP_TRAJ_LEN = 100
DECAY_FACTOR = 0.9

cls_model_path = r'scene_understanding/classifier.pkl'

parser = argparse.ArgumentParser('--- Trajectory clustering and path modeling ---')
parser.add_argument('--log_dir', default=None,
                    help='directory to save trajctory clustering models')
args = parser.parse_args()

if __name__ == '__main__':
    # If you want to receive string data from redis directly, you can set "decode_responses" to True
    decode_responses = False
    receiver = receiver.ReceiverUtils(
        host='192.168.202.90',
        password='pcl1305',
        decode_responses=decode_responses)
    sender = sender.SenderUtilser(
        host='192.168.202.90',
        password='pcl1305',
        auto_delete_limit=10)

    scene.init(MAX_NUM_CLUSTER_TRAJS, ABNORMAL_PERCENT, MIN_REP_TRAJ_LEN, DECAY_FACTOR)
    while True:
        try:
            QUEUE_KEY = 'trajs'
            ret_list = receiver.reveive_data(key=QUEUE_KEY, limit_count=1000)
            if not ret_list:
                time.sleep(10)
                continue

            trajs = scene.TrajVec()
            # If your list item is json like object, you can format the list item into dictionary.
            for item in ret_list:
                if not decode_responses:
                    # If you set "decode_responses" to False, we should decode the item in terms of
                    # the format of item is byte rather str.
                    traj = scene.Traj.decode(item)
                else:
                    raise NotImplementedError
                trajs.append((traj.track_id, traj.rects))

            if args.log_dir is None:
                model_path = '/tmp/trafficUAC/'
            else:
                model_path = args.log_dir
            time_str = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
            model_path = os.path.join(model_path, time_str)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            scene.train(trajs, MAX_ITER, model_path, W, H)
            routes = scene.generate()
            routes_py = []
            for route in routes:
                traj = []
                for j in range(route.num_landmark()):
                    traj.append(route.get_landmark(j).x)
                    traj.append(route.get_landmark(j).y)
                pre = predict.predict([traj], cls_model_path, sample_num=21)
                traj.clear()
                # The id is reserved for cluster
                # route.id = pre
                routes_py.append(scene.Route(route))

            QUEUE_KEY = 'routes'
            sender.send_with_byte(key=QUEUE_KEY, data_byte=scene.encode_routes(routes_py))

        except (KeyboardInterrupt, SystemExit):
            scene.destroy()
            raise
