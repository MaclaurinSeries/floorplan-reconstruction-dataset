import numpy as np

__all__ = ['getRoomID', 'getRoomName', 'getBoundaryID', 'getBoundaryName', 'getIconID', 'getIconName', 'oneHotRoomID', 'oneHotBoundaryID', 'oneHotIconID', 'lenIconBoundary', 'lenRoom']

rooms = ['Background', # 0
 'Outdoor',
 'Wall', # 2
 'Kitchen',
 'Dining',
 'Bedroom',
 'Bath',
 'Entry',
 'Railing', # 8
 'Storage',
 'Garage',
 'Room',
 'LivingRoom']

# sebelumnya
# none, window, door, closet, ....

# perubahan, index icon bakal dikurangi 3

icons = [
 'Door',
 'Window',
 'Closet',
 'ElectricalAppliance',
 'Toilet',
 'Sink',
 'SaunaBench',
 'Fireplace',
 'Bathtub',
 'Chimney',
 'Stairs']

boundaries = [
    'Door',
    'Wall',
    'Window'
]

boundaries_selected = {
    'Wall': 1,
    'Window': 2,
    'Door': 0
}

all_rooms = {"Background": 0,
             "Alcove": 1,
             "Attic": 2,
             "Ballroom": 3,
             "Bar": 4,
             "Basement": 5,
             "Bath": 6,
             "Bedroom": 7,
             "Below150cm": 8,
             "CarPort": 9,
             "Church": 10,
             "Closet": 11,
             "ConferenceRoom": 12,
             "Conservatory": 13,
             "Counter": 14,
             "Den": 15,
             "Dining": 16,
             "DraughtLobby": 17,
             "DressingRoom": 18,
             "EatingArea": 19,
             "Elevated": 20,
             "Elevator": 21,
             "Entry": 22,
             "ExerciseRoom": 23,
             "Garage": 24,
             "Garbage": 25,
             "Hall": 26,
             "HallWay": 27,
             "HotTub": 28,
             "Kitchen": 29,
             "Library": 30,
             "LivingRoom": 31,
             "Loft": 32,
             "Lounge": 33,
             "MediaRoom": 34,
             "MeetingRoom": 35,
             "Museum": 36,
             "Nook": 37,
             "Office": 38,
             "OpenToBelow": 39,
             "Outdoor": 40,
             "Pantry": 41,
             "Reception": 42,
             "RecreationRoom": 43,
             "RetailSpace": 44,
             "Room": 45,
             "Sanctuary": 46,
             "Sauna": 47,
             "ServiceRoom": 48,
             "ServingArea": 49,
             "Skylights": 50,
             "Stable": 51,
             "Stage": 52,
             "StairWell": 53,
             "Storage": 54,
             "SunRoom": 55,
             "SwimmingPool": 56,
             "TechnicalRoom": 57,
             "Theatre": 58,
             "Undefined": 59,
             "UserDefined": 60,
             "Utility": 61,
             "Wall": 62,
             "Railing": 63,
             "Stairs": 64}

rooms_selected = {"Alcove": 11,
                  "Attic": 11,
                  "Ballroom": 11,
                  "Bar": 11,
                  "Basement": 11,
                  "Bath": 6,
                  "Bedroom": 5,
                  "CarPort": 10,
                  "Church": 11,
                  "Closet": 9,
                  "ConferenceRoom": 11,
                  "Conservatory": 11,
                  "Counter": 11,
                  "Den": 11,
                  "Dining": 4,
                  "DraughtLobby": 7,
                  "DressingRoom": 9,
                  "EatingArea": 4,
                  "Elevated": 11,
                  "Elevator": 11,
                  "Entry": 7,
                  "ExerciseRoom": 11,
                  "Garage": 10,
                  "Garbage": 11,
                  "Hall": 11,
                  "HallWay": 7,
                  "HotTub": 11,
                  "Kitchen": 3,
                  "Library": 11,
                  "LivingRoom": 12,
                  "Loft": 11,
                  "Lounge": 12,
                  "MediaRoom": 11,
                  "MeetingRoom": 11,
                  "Museum": 11,
                  "Nook": 11,
                  "Office": 11,
                  "OpenToBelow": 11,
                  "Outdoor": 1,
                  "Pantry": 11,
                  "Reception": 11,
                  "RecreationRoom": 11,
                  "RetailSpace": 11,
                  "Room": 11,
                  "Sanctuary": 11,
                  "Sauna": 6,
                  "ServiceRoom": 11,
                  "ServingArea": 11,
                  "Skylights": 11,
                  "Stable": 11,
                  "Stage": 11,
                  "StairWell": 11,
                  "Storage": 9,
                  "SunRoom": 11,
                  "SwimmingPool": 11,
                  "TechnicalRoom": 11,
                  "Theatre": 11,
                  "Undefined": 11,
                  "UserDefined": 11,
                  "Utility": 11,
                  "Background": 0,
                  "Wall": 2,
                  "Railing": 8}

room_name_map = {"Alcove": "Room",
                 "Attic": "Room",
                 "Ballroom": "Room",
                 "Bar": "Room",
                 "Basement": "Room",
                 "Bath": "Bath",
                 "Bedroom": "Bedroom",
                 "Below150cm": "Room",
                 "CarPort": "Garage",
                 "Church": "Room",
                 "Closet": "Storage",
                 "ConferenceRoom": "Room",
                 "Conservatory": "Room",
                 "Counter": "Room",
                 "Den": "Room",
                 "Dining": "Dining",
                 "DraughtLobby": "Entry",
                 "DressingRoom": "Storage",
                 "EatingArea": "Dining",
                 "Elevated": "Room",
                 "Elevator": "Room",
                 "Entry": "Entry",
                 "ExerciseRoom": "Room",
                 "Garage": "Garage",
                 "Garbage": "Room",
                 "Hall": "Room",
                 "HallWay": "Entry",
                 "HotTub": "Room",
                 "Kitchen": "Kitchen",
                 "Library": "Room",
                 "LivingRoom": "LivingRoom",
                 "Loft": "Room",
                 "Lounge": "LivingRoom",
                 "MediaRoom": "Room",
                 "MeetingRoom": "Room",
                 "Museum": "Room",
                 "Nook": "Room",
                 "Office": "Room",
                 "OpenToBelow": "Room",
                 "Outdoor": "Outdoor",
                 "Pantry": "Room",
                 "Reception": "Room",
                 "RecreationRoom": "Room",
                 "RetailSpace": "Room",
                 "Room": "Room",
                 "Sanctuary": "Room",
                 "Sauna": "Bath",
                 "ServiceRoom": "Room",
                 "ServingArea": "Room",
                 "Skylights": "Room",
                 "Stable": "Room",
                 "Stage": "Room",
                 "StairWell": "Room",
                 "Storage": "Storage",
                 "SunRoom": "Room",
                 "SwimmingPool": "Room",
                 "TechnicalRoom": "Room",
                 "Theatre": "Room",
                 "Undefined": "Room",
                 "UserDefined": "Room",
                 "Utility": "Room",
                 "Wall": "Wall",
                 "Railing": "Railing",
                 "Background": "Background"}  # Not in data. The default outside label

all_icons = {"Empty": 0,
             "Window": 1,
             "Door": 2,
             "BaseCabinet": 3,
             "BaseCabinetRound": 4,
             "BaseCabinetTriangle": 5,
             "Bathtub": 6,
             "BathtubRound": 7,
             "Chimney": 8,
             "Closet": 9,
             "ClosetRound": 10,
             "ClosetTriangle": 11,
             "CoatCloset": 12,
             "CoatRack": 13,
             "CornerSink": 14,
             "CounterTop": 15,
             "DoubleSink": 16,
             "DoubleSinkRight": 17,
             "ElectricalAppliance": 18,
             "Fireplace": 19,
             "FireplaceCorner": 20,
             "FireplaceRound": 21,
             "GasStove": 22,
             "Housing": 23,
             "Jacuzzi": 24,
             "PlaceForFireplace": 25,
             "PlaceForFireplaceCorner": 26,
             "PlaceForFireplaceRound": 27,
             "RoundSink": 28,
             "SaunaBenchHigh": 29,
             "SaunaBenchLow": 30,
             "SaunaBenchMid": 31,
             "Shower": 32,
             "ShowerCab": 33,
             "ShowerScreen": 34,
             "ShowerScreenRoundLeft": 35,
             "ShowerScreenRoundRight": 36,
             "SideSink": 37,
             "Sink": 38,
             "Toilet": 39,
             "Urinal": 40,
             "WallCabinet": 41,
             "WaterTap": 42,
             "WoodStove": 43,
             "Misc": 44,
             "SaunaBench": 45,
             "SaunaStove": 46,
             "WashingMachine": 47,
             "IntegratedStove": 48,
             "Dishwasher": 49,
             "GeneralAppliance": 50,
             "ShowerPlatform": 51,
             "Stairs": 11}

icons_selected = {
    "Closet": 0,
    "ClosetRound": 0,
    "ClosetTriangle": 0,
    "CoatCloset": 0,
    "CoatRack": 0,
    "CounterTop": 0,
    "Housing": 0,
    "ElectricalAppliance": 1,
    "WoodStove": 1,
    "GasStove": 1,
    "SaunaStove": 1,
    "Toilet": 2,
    "Urinal": 2,
    "SideSink": 3,
    "Sink": 3,
    "RoundSink": 3,
    "CornerSink": 3,
    "DoubleSink": 3,
    "DoubleSinkRight": 3,
    "WaterTap": 3,
    "SaunaBenchHigh": 4,
    "SaunaBenchLow": 4,
    "SaunaBenchMid": 4,
    "SaunaBench": 4,
    "Fireplace": 5,
    "FireplaceCorner": 5,
    "FireplaceRound": 5,
    "PlaceForFireplace": 5,
    "PlaceForFireplaceCorner": 5,
    "PlaceForFireplaceRound": 5,
    "Bathtub": 6,
    "BathtubRound": 6,
    "Chimney": 7,
    "Misc": None,
    "BaseCabinetRound": None,
    "BaseCabinetTriangle": None,
    "BaseCabinet": None,
    "WallCabinet": None,
    "Shower": None,
    "ShowerCab": None,
    "ShowerPlatform": None,
    "ShowerScreen": None,
    "ShowerScreenRoundRight": None,
    "ShowerScreenRoundLeft": None,
    "Jacuzzi": None,
    "WashingMachine": None,
    "IntegratedStove": 1,
    "Dishwasher": 1,
    "GeneralAppliance": 1,
    "Stairs": 8,
    None: None
}

icon_name_map = {"Closet": "Closet",
                  "ClosetRound": "Closet",
                  "ClosetTriangle": "Closet",
                  "CoatCloset": "Closet",
                  "CoatRack": "Closet",
                  "CounterTop": "Closet",
                  "Housing": "Closet",
                  "ElectricalAppliance": "ElectricalAppliance",
                  "WoodStove": "ElectricalAppliance",
                  "GasStove": "ElectricalAppliance",
                  "SaunaStove": "ElectricalAppliance",
                  "Toilet": "Toilet",
                  "Urinal": "Toilet",
                  "SideSink": "Sink",
                  "Sink": "Sink",
                  "RoundSink": "Sink",
                  "CornerSink": "Sink",
                  "DoubleSink": "Sink",
                  "DoubleSinkRight": "Sink",
                  "WaterTap": "Sink",
                  "SaunaBenchHigh": "SaunaBench",
                  "SaunaBenchLow": "SaunaBench",
                  "SaunaBenchMid": "SaunaBench",
                  "SaunaBench": "SaunaBench",
                  "Fireplace": "Fireplace",
                  "FireplaceCorner": "Fireplace",
                  "FireplaceRound": "Fireplace",
                  "PlaceForFireplace": "Fireplace",
                  "PlaceForFireplaceCorner": "Fireplace",
                  "PlaceForFireplaceRound": "Fireplace",
                  "Bathtub": "Bathtub",
                  "BathtubRound": "Bathtub",
                  "Chimney": "Chimney",
                  "Misc": None,
                  "BaseCabinetRound": None,
                  "BaseCabinetTriangle": None,
                  "BaseCabinet": None,
                  "WallCabinet": None,
                  "Shower": None,
                  "ShowerCab": None,
                  "ShowerPlatform": None,
                  "ShowerScreen": None,
                  "ShowerScreenRoundRight": None,
                  "ShowerScreenRoundLeft": None,
                  "Jacuzzi": None,
                  "WashingMachine": None,
                  "IntegratedStove": "ElectricalAppliance",
                  "Dishwasher": "ElectricalAppliance",
                  "GeneralAppliance": "ElectricalAppliance",
                  "Stairs": "Stairs",
                  None: None}

def getRoomID(room_name):
    return rooms_selected[room_name_map[room_name]]

def getIconID(icon_name):
    return icons_selected[icon_name_map[icon_name]]

def getRoomName(room):
    if (isinstance(room, int)):
        return rooms[room]
    else:
        return room_name_map[room]

def getIconName(icon):
    if (isinstance(icon, int)):
        return icons[icon]
    else:
        return icon_name_map[icon]

def getBoundaryName(boundary):
    if (isinstance(boundary, int)):
        return boundaries[boundary]
    elif (boundary in boundaries):
        return boundary
    else:
        return None
    
def getBoundaryID(boundary_name):
    return boundaries_selected[boundary_name]

def oneHotRoomID(room_ID):
    room_len = len(rooms)
    label = [.0] * room_len
    label[room_ID] = 1.0

    return label

def oneHotBoundaryID(boundary_ID, with_icon=False):
    icon_len = len(icons)
    boundary_len = len(boundaries)
    label = [.0] * boundary_len
    label[boundary_ID] = 1.0

    if with_icon:
        label = label + [.0] * icon_len
    
    return label

def oneHotIconID(icon_ID, with_boundary=False):
    icon_len = len(icons)
    boundary_len = len(boundaries)
    label = [.0] * icon_len
    label[icon_ID] = 1.0

    if with_boundary:
        label = [.0] * boundary_len + label
    
    return label

def lenIconBoundary():
    return len(icons) + len(boundaries)

def lenBoundary():
    return len(boundaries)

def lenIcon():
    return len(icons)

def lenRoom():
    return len(rooms)