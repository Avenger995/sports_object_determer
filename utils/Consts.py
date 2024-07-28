from utils.DrawUtil import Color
import numpy as np

# black
PLAYER_COLOR_BOUNDARY = [(0, 0, 0), (91, 91, 91)]

# red
RESERVE_PLAYER_COLOR_BOUNDARY = [(150, 36, 36), (244, 204, 204)]

BOUNDARIES = [PLAYER_COLOR_BOUNDARY] + [RESERVE_PLAYER_COLOR_BOUNDARY]

# white
BALL_COLOR_HEX = "#FFFFFF"
BALL_COLOR = Color.from_hex_string(BALL_COLOR_HEX)

# red
GOALKEEPER_COLOR_HEX = "#850101"
GOALKEEPER_COLOR = Color.from_hex_string(GOALKEEPER_COLOR_HEX)

# pink
PLAYER_COLOR_HEX = "#C27BA0"
PLAYER_COLOR = Color.from_hex_string(PLAYER_COLOR_HEX)

# purple
RESERVE_PLAYER_COLOR_HEX = "#8E7CC3"
RESERVE_PLAYER_COLOR = Color.from_hex_string(RESERVE_PLAYER_COLOR_HEX)

# yellow
REFEREE_COLOR_HEX = "#FFFF00"
REFEREE_COLOR = Color.from_hex_string(REFEREE_COLOR_HEX)


COLORS = [
    BALL_COLOR,
    GOALKEEPER_COLOR,
    PLAYER_COLOR,
    RESERVE_PLAYER_COLOR,
    REFEREE_COLOR
]
THICKNESS = 4


# black
MARKER_CONTOUR_COLOR_HEX = "000000"
MARKER_CONTOUR_COLOR = Color.from_hex_string(MARKER_CONTOUR_COLOR_HEX)

# red
PLAYER_MARKER_FILL_COLOR_HEX = "FF0000"
PLAYER_MARKER_FILL_COLOR = Color.from_hex_string(PLAYER_MARKER_FILL_COLOR_HEX)

# green
BALL_MERKER_FILL_COLOR_HEX = "00FF00"
BALL_MARKER_FILL_COLOR = Color.from_hex_string(BALL_MERKER_FILL_COLOR_HEX)

MARKER_CONTOUR_THICKNESS = 2
MARKER_WIDTH = 20
MARKER_HEIGHT = 20
MARKER_MARGIN = 10

# distance in pixels from the player's bounding box where we consider the ball is in his possession
PLAYER_IN_POSSESSION_PROXIMITY = 30

# point to draw text of possession
POSSESSION_POINT_MAIN = (30, 30)
POSSESSION_POINT_RESERVE = (30, 50)

# enum team
TEAM1 = 1
TEAM2 = 2

# points for pixel mapper
SOURCE_POINTS = [[500, 250], [1, 970], [2550, 450], [2080, 1430]]
MAP_POINTS = [[13, 14], [67, 450], [401, 15], [332, 450]]

ARRAY_SOURCE_POINTS = np.float32(SOURCE_POINTS)
ARRAY_MAP_POINTS = np.float32(MAP_POINTS)

MAP_HEIGHT = 345
MAP_WIDTH = 500

SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080
