# Vertical Descent Environment

This environment is based on BlueSky-Gym's `VerticalDescent` environment, which simulates a vertical descent scenario.
The goal of the environment is to stay at the target altitude for as long as possible before initiating the descent towards the runway. The agent controls the vertical velocity of the aircraft.

![image](./example.gif)

## Action Space
The action space consists of a single continuous action representing the vertical velocity of the aircraft. The action is bounded between -1 and 1, where -1 represents maximum descent and 1 represents maximum ascent.

## Observation Space
The observation space consists of the following elements:
- `altitude`: The current altitude of the aircraft.
- `vertical_velocity`: The current vertical velocity of the aircraft.
- `target_altitude`: The target altitude for the aircraft.
- `runway_distance`: The distance to the runway.

All observations are **continuous** values. The observation is presented as a Dictionary with the keys: `altitude`, `vz`, `target_altitude`, and `runway_distance` with the corresponding values.