import sys

import scene
from transfer import sender


if __name__ == '__main__':
    traj_file = sys.argv[1]
    rects, W, H = scene.get_all_raw_traj(traj_file)

    QUEUE_KEY = 'trajs'
    sender = sender.SenderUtilser(
        host='192.168.202.90',
        password='pcl1305',
        auto_delete_limit=2000)

    track_id = 0
    for rect in rects:
        traj = scene.Traj(track_id, rect)
        track_id += 1
        sender.send_with_byte(key=QUEUE_KEY, data_byte=traj.encode())
