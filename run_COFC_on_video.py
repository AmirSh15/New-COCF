import time
import numpy as np
from argparse import ArgumentParser
import cv2
# from matplotlib import pyplot as plt
from utils.cluster_utils import ClustersShots, ClustersTracks
from utils.cofc_utils import extract_bboxes_and_features, initialize_deep_models, shot_boundary



def get_facetracks_and_links(shot_data, th_feats=1.0, th_overlap=0.85):
    t1 = time.time()
    fno = 0
    nFaces = len(shot_data)
    ls_inds = [0]

    ls_cannotlink = []
    cl_tracks = ClustersTracks(simThresh=th_overlap, featThresh=th_feats)

    for i in range(1, nFaces):

        fno = shot_data[i].fno
        if (fno != shot_data[i - 1].fno):
            # do something
            ls_data = [shot_data[x] for x in ls_inds]
            ls_cl = cl_tracks.cluster_online(ls_data)
            ls_cannotlink.append(ls_cl)
            ls_inds = [i]
        else:
            ls_inds.append(i)

    ls_tracks = cl_tracks.clusters
    l = len(ls_tracks)
    qMatrix = np.ones((l, l))

    for l in ls_cannotlink:
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                qMatrix[i, j] = 0
                qMatrix[j, i] = 0
    t2 = time.time()

    # delete tracks with length < 15
    i = 0
    while (i < len(ls_tracks)):
        if (len(ls_tracks[i]) < 15):
            del ls_tracks[i]
            qMatrix = np.delete(qMatrix, i, axis=0)
            qMatrix = np.delete(qMatrix, i, axis=1)
            i -= 1
        i += 1
    return ls_tracks, qMatrix


def process_shot(clusters_shot, ls_frames, model, th_feats, th_overlap):
    t1 = time.time()
    shot_data = extract_bboxes_and_features(ls_frames, model)  # list of face_element
    t2 = time.time()
    print("Shot has %d faces on which MXNet took %.3f secs" % (len(shot_data), t2 - t1))
    t1 = t2
    face_tracks, qMatrix = get_facetracks_and_links(shot_data, th_feats, th_overlap)
    clusters_shot.cluster_online(face_tracks, qMatrix)  # the functionwa in paperwa
    t2 = time.time()
    print("Processing the shot and clustering took %.2f secs" % (t2 - t1))
    return face_tracks, qMatrix


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-vp", dest="vid_path", help="Path to the video file", default='./videos/1.mp4',
                        type=str)
    parser.add_argument("-sd", dest="save_dir", help="Directory path for saving the output",
                        default="./Clusters", type=str)
    parser.add_argument("-ft", dest="feat_thresh",
                        help="Threshold of distance bw features to belong to different persons",
                        default="1.0", type=float)
    parser.add_argument("-ot", dest="overlap_thresh",
                        help="Threshold of overlap above which two faces in consecutive frames "
                             "will belong to same track",
                        default="0.90", type=float)
    parser.add_argument("-st", dest="sim_thresh", help="Threshold of Similarity for facetracks "
                                                       "to belong to a cluster",
                        default="2.8", type=float)
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='./pretrained_models/model-r34-amf,0',
                        help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

    args = parser.parse_args()

    path = args.vid_path
    simThreshShot = args.sim_thresh
    th_feats = args.feat_thresh
    th_overlap = args.overlap_thresh
    saveDir = args.save_dir

    cap = cv2.VideoCapture(path)
    model = initialize_deep_models(args)

    ret = False

    ls_frames = []

    for i in range(2):
        ret, frame = cap.read()
        assert (ret == True)
        if (i == 0):
            ppframe = frame
        if (i == 1):
            pframe = frame
        ls_frames.append((i, frame))

    clusters_shot = ClustersShots(simThreshShot, saveDir)
    kk = 2
    ft = []
    while (ret == True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (ret == False):
            # No frame detected hence video is ended.
            print("Processing shot of n frames: " + str(len(ls_frames)))
            ft, qmat = process_shot(clusters_shot, ls_frames, model, th_feats, th_overlap)
            print(qmat)
            ls_frames = []
        else:
            ls_frames.append((kk, frame))
            sb = shot_boundary(ppframe, pframe, frame)
            # If shot boundary is detected or clip is more than 100 frames (assuming framerate ~20-30fps), process it
            if (sb or len(ls_frames) > 24 * 10):  # more than 10s
                print("Processing shot of n frames: " + str(len(ls_frames)))
                ft, qmat = process_shot(clusters_shot, ls_frames, model, th_feats, th_overlap)
                print(qmat)
                ls_frames = []
            # update prev-prev frame and prev-frame
            ppframe = pframe
            pframe = frame
            kk += 1

    print("Completed for the video: " + path)

