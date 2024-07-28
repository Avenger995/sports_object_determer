from __future__ import annotations
import torch
from ultralytics import YOLO
import sys

from Settings import Settings
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import os
from tqdm import tqdm

from stats_collectors.PassCollector import PassCollector
from utils import (BYTETrackerArgs, Detection, BaseAnnotator, Consts, MarkerAnnotator, TextAnnotator, VideoUtil)
from stats_collectors import PossessionService
from utils.Detection import filter_detections_by_class, detections2boxes, match_detections_with_tracks
from utils.DrawUtil import Color
from utils.MarkerAnnotator import get_player_in_possession
from utils.VideoUtil import get_video_writer, generate_frames
from utils.PixelMapper import PixelMapper
import cv2
from stats_collectors.TouchCollector import TouchCollector


def main(settings: Settings):

    HOME = os.getcwd()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    sys.path.append(f"C:/Users/Gubay/OneDrive/Documents/v3/ByteTrack")
    WEIGHT_PATH = settings.model_path

    model = YOLO(WEIGHT_PATH).to(device)

    SOURCE_VIDEO_PATH = settings.file_path
    TARGET_VIDEO_PATH = f"C:/Users/Gubay/OneDrive/Documents/v3/videos/resultv2/result.mp4"
    MAP_PICTURE = f"C:/Users/Gubay/OneDrive/Documents/v3/static/pitch.png"
    TOUCH_TARGET_PATH = f"C:/Users/Gubay/OneDrive/Documents/v3/videos/resultv2/touches_result.png"
    PASS_TARGET_PATH = f"C:/Users/Gubay/OneDrive/Documents/v3/videos/resultv2/pass_result.png"

    pm = PixelMapper(settings.source_points, settings.map_points)

    video_config = VideoUtil.VideoConfig(
        fps=30,
        width=Consts.SOURCE_WIDTH,
        height=Consts.SOURCE_HEIGHT)
    video_writer = get_video_writer(
        target_video_path=TARGET_VIDEO_PATH,
        video_config=video_config)

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))

    # initiate annotators
    base_annotator = BaseAnnotator.BaseAnnotator(
        colors=[
            Consts.BALL_COLOR,
            Consts.PLAYER_COLOR,
            Consts.PLAYER_COLOR,
            Consts.RESERVE_PLAYER_COLOR,
            Consts.REFEREE_COLOR
        ],
        thickness=Consts.THICKNESS,
        pm=pm)

    player_goalkeeper_text_annotator = TextAnnotator.TextAnnotator(
        Consts.PLAYER_COLOR, text_color=Color(255, 255, 255), text_thickness=2,
        reserve_background_color=Consts.RESERVE_PLAYER_COLOR, pm=pm)
    referee_text_annotator = TextAnnotator.TextAnnotator(
        Consts.REFEREE_COLOR, text_color=Color(0, 0, 0), text_thickness=2, pm=pm)

    ball_marker_annotator = MarkerAnnotator.MarkerAnnotator(
        color=Consts.BALL_MARKER_FILL_COLOR, pm=pm, is_ball=True)
    player_in_possession_marker_annotator = MarkerAnnotator.MarkerAnnotator(
        color=Consts.PLAYER_MARKER_FILL_COLOR, pm=pm)
    player_marker_annotator = MarkerAnnotator.MarkerAnnotator(color=Consts.PLAYER_MARKER_FILL_COLOR, pm=pm)

    possession = PossessionService.PossessionService(color_main=Consts.PLAYER_COLOR,
                                                     color_reserved=Consts.RESERVE_PLAYER_COLOR)

    matrix = cv2.getPerspectiveTransform(settings.source_points, settings.map_points)
    pitch = cv2.resize(cv2.imread(MAP_PICTURE), (Consts.MAP_WIDTH, Consts.MAP_HEIGHT), interpolation=cv2.INTER_AREA)

    touch_collector = TouchCollector(
        pm=pm,
        colors=[Consts.PLAYER_COLOR, Consts.RESERVE_PLAYER_COLOR])
    pass_collector = PassCollector(
        pm=pm,
        colors=[Consts.PLAYER_COLOR, Consts.RESERVE_PLAYER_COLOR])

    # initiate tracker
    byte_tracker = BYTETracker(BYTETrackerArgs.BYTETrackerArgs())

    # loop over frames
    for frame in tqdm(frame_iterator, total=750):
        # run detector
        results = model(frame)[0]
        detections = Detection.Detection.from_results(
            pred=results.boxes.data.cpu().numpy(),
            names=model.names,
            boundaries=settings.boundaries,
            frame=frame)

        map = cv2.warpPerspective(frame, matrix, (Consts.MAP_WIDTH, Consts.MAP_HEIGHT))
        map = cv2.addWeighted(pitch, 1, map, 0, 0)

        # filter detections by class
        ball_detections = filter_detections_by_class(detections=detections, class_name="ball")
        referee_detections = filter_detections_by_class(detections=detections, class_name="referee")
        goalkeeper_detections = filter_detections_by_class(detections=detections, class_name="goalkeeper")
        player_detections = filter_detections_by_class(detections=detections, class_name="player")

        player_goalkeeper_detections = player_detections + goalkeeper_detections
        tracked_detections = player_detections + goalkeeper_detections

        # calculate player in possession
        player_in_possession_detection = get_player_in_possession(
            player_detections=player_goalkeeper_detections,
            ball_detections=ball_detections,
            proximity=Consts.PLAYER_IN_POSSESSION_PROXIMITY)

        # track
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=tracked_detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracked_detections = match_detections_with_tracks(detections=tracked_detections, tracks=tracks)

        tracked_referee_detections = filter_detections_by_class(detections=tracked_detections, class_name="referee")
        tracked_goalkeeper_detections = filter_detections_by_class(detections=tracked_detections, class_name="goalkeeper")
        tracked_player_detections = filter_detections_by_class(detections=tracked_detections, class_name="player")

        # annotate video frame
        annotated_image = frame.copy()
        annotated_image = base_annotator.annotate(
            image=annotated_image,
            detections=tracked_detections)

        annotated_image = player_goalkeeper_text_annotator.annotate(
            image=annotated_image,
            detections=tracked_goalkeeper_detections + tracked_player_detections)

        player_in_possession_detection = [player_in_possession_detection] if player_in_possession_detection else []

        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image,
            detections=ball_detections)
        annotated_image = player_marker_annotator.annotate(
            image=annotated_image,
            detections=player_in_possession_detection)

        touch_collector.append(player_in_possession_detection)
        pass_collector.append(player_in_possession_detection)

        annotated_image = possession.annotate(
            image=annotated_image,
            detections=player_in_possession_detection)

        map = base_annotator.annotate_map(
            image=map,
            detections=tracked_detections)
        map = player_goalkeeper_text_annotator.annotate_map(
            image=map,
            detections=tracked_detections)

        map = ball_marker_annotator.annotate_map(
            image=map,
            detections=ball_detections,
            radius=5,
            fill=cv2.FILLED)
        map = player_marker_annotator.annotate_map(
            image=map,
            detections=player_in_possession_detection)

        x_offset = Consts.SOURCE_WIDTH - Consts.MAP_WIDTH
        y_offset = Consts.SOURCE_HEIGHT - Consts.MAP_HEIGHT
        annotated_image[y_offset:y_offset + map.shape[0], x_offset:x_offset + map.shape[1]] = map

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()

    cv2.imwrite(TOUCH_TARGET_PATH, touch_collector.get_image(pitch))
    cv2.imwrite(PASS_TARGET_PATH, pass_collector.get_image(pitch))


if __name__ == "__main__":
    file_path = str(sys.argv[1])
    model_path = str(sys.argv[2])
    source_points = sys.argv[3]
    map_points = sys.argv[4]
    boundaries = sys.argv[5]
    settings = Settings(file_path, model_path, source_points, map_points, boundaries)
    main(settings)
